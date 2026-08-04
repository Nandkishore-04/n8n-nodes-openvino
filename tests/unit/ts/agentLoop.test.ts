import { runAgent } from '../../../nodes/OpenVinoAgent/loop';
import { toolRegistry, validateMath, validateTotals, wordsToNumber, parseAmountWords, coverageNote, buildTools, ToolContext } from '../../../nodes/OpenVinoAgent/tools/builtIn';

// A fake chat model that returns a scripted sequence of replies.
function scriptedChat(replies: string[]) {
	let i = 0;
	return async () => replies[Math.min(i++, replies.length - 1)];
}

describe('runAgent — prompt-based loop', () => {
	it('calls a tool, feeds the result back, then returns the final answer', async () => {
		const chat = scriptedChat([
			'{"tool":"validate_math","args":{"subtotal":1100,"tax":134.56,"total":1234.56}}',
			'{"final":"{\\"decision\\":\\"enriched\\"}"}',
		]);
		const res = await runAgent({ chat, tools: toolRegistry(), system: 'sys', input: 'invoice' });

		expect(res.incomplete).toBe(false);
		expect(res.iterations[0].tool).toBe('validate_math');
		expect((res.iterations[0].result as any).valid).toBe(true);
		expect(res.final).toContain('enriched');
	});

	it('returns the final answer directly when the model is done', async () => {
		const chat = scriptedChat(['{"final":"all good"}']);
		const res = await runAgent({ chat, tools: toolRegistry(), system: 's', input: 'x' });
		expect(res.final).toBe('all good');
		expect(res.iterations).toHaveLength(1);
	});

	it('surfaces decision + finalData when final is an object (no [object Object])', async () => {
		const chat = scriptedChat(['{"final":{"decision":"flagged","reason":"unreadable"}}']);
		const res = await runAgent({ chat, tools: toolRegistry(), system: 's', input: 'x' });
		expect(res.final).not.toContain('[object Object]');
		expect(res.decision).toBe('flagged');
		expect((res.finalData as any).reason).toBe('unreadable');
	});

	it('surfaces reason + confidence from a structured final', async () => {
		const chat = scriptedChat([
			'{"final":{"decision":"enriched","reason":"new invoice","confidence":0.92,"document_type":"invoice"}}',
		]);
		const res = await runAgent({ chat, tools: toolRegistry(), system: 's', input: 'x' });
		expect(res.decision).toBe('enriched');
		expect(res.reason).toBe('new invoice');
		expect(res.confidence).toBe(0.92);
	});

	it('strips <think> blocks before parsing', async () => {
		const chat = scriptedChat(['<think>reasoning here</think>\n{"final":"done"}']);
		const res = await runAgent({ chat, tools: toolRegistry(), system: 's', input: 'x' });
		expect(res.final).toBe('done');
	});

	it('reports a clear error for an unknown tool but keeps going', async () => {
		const chat = scriptedChat([
			'{"tool":"does_not_exist","args":{}}',
			'{"final":"recovered"}',
		]);
		const res = await runAgent({ chat, tools: toolRegistry(), system: 's', input: 'x' });
		expect((res.iterations[0].result as any).error).toContain('unknown tool');
		expect(res.final).toBe('recovered');
	});

	it('flags incomplete when it hits the iteration cap', async () => {
		const chat = scriptedChat(['{"tool":"validate_math","args":{}}']); // never finishes
		const res = await runAgent({ chat, tools: toolRegistry(), system: 's', input: 'x', maxIters: 3 });
		expect(res.incomplete).toBe(true);
		expect(res.iterations.length).toBe(3);
	});

	it('retries once then flags a persistent degenerate runaway', async () => {
		const runaway = '{"tool":"extract_fields","args":{"schema":{' + '"a_employment_supervisor":"str",'.repeat(300);
		const chat = scriptedChat([runaway]); // always degenerate → retry, then flag
		const res = await runAgent({ chat, tools: toolRegistry(), system: 's', input: 'x' });
		expect(res.decision).toBe('flagged');
		expect(res.incomplete).toBe(false);
		expect(res.iterations).toHaveLength(2); // one retry, then flagged (not 8 loops)
	});
	it('recovers when a degenerate turn is followed by a clean one', async () => {
		const runaway = 'x'.repeat(4000); // degenerate first
		const chat = scriptedChat([runaway, '{"final":"ok"}']); // clean on retry
		const res = await runAgent({ chat, tools: toolRegistry(), system: 's', input: 'x' });
		expect(res.final).toBe('ok'); // retry succeeded, no false flag
	});
});

describe('buildTools — all 9 tools', () => {
	const baseCtx = (over: Partial<ToolContext> = {}): ToolContext => ({
		chat: async () => '{"vendor":"Acme"}',
		http: async () => ({}),
		gatewayUrl: 'http://gateway:8000',
		noThink: true,
		...over,
	});

	it('registers all eight tools', () => {
		const tools = buildTools(baseCtx());
		expect([...tools.keys()].sort()).toEqual(
			['check_duplicate', 'extract_fields', 'flag_for_review', 'knowledge_search', 'lookup_vendor', 'recall', 'retry_document_extraction', 'validate_math', 'validate_totals'],
		);
	});

	it('extract_fields parses the LLM JSON reply', async () => {
		const tools = buildTools(baseCtx());
		const r = await tools.get('extract_fields')!.run({ schema: { vendor: 'str' }, text: 'INVOICE Acme' }) as any;
		expect(r.vendor).toBe('Acme');
	});

	it('retry_document_extraction errors clearly with no source file', async () => {
		const r = await buildTools(baseCtx()).get('retry_document_extraction')!.run({ enhance: true }) as any;
		expect(r.error).toContain('no source document');
	});

	it('retry calls the gateway when a source file is present', async () => {
		let called: any;
		const tools = buildTools(baseCtx({
			sourceFile: { dataB64: 'abc', filename: 'x.pdf' },
			http: async (opts) => { called = opts; return { text: 'better', confidence: 0.9, source: 'ocr' }; },
		}));
		const r = await tools.get('retry_document_extraction')!.run({ enhance: true, dpi: 300 }) as any;
		expect(called.url).toContain('/v1/document/infer');
		expect(called.body.dpi).toBe(300);
		expect(r.text).toBe('better');
	});

	it('check_duplicate / lookup_vendor / recall degrade gracefully when unconfigured', async () => {
		const tools = buildTools(baseCtx());
		expect((await tools.get('check_duplicate')!.run({ document_id: 'X' }) as any).note).toContain('no dedup backend');
		expect((await tools.get('lookup_vendor')!.run({ name: 'X' }) as any).note).toContain('no vendor backend');
		expect((await tools.get('recall')!.run({ query: 'X' }) as any).note).toContain('no memory configured');
		expect((await tools.get('knowledge_search')!.run({ query: 'X' }) as any).note).toContain('no Qdrant configured');
	});
});

describe('validate_math tool', () => {
	it('passes when subtotal + tax == total', async () => {
		expect((await validateMath.run({ subtotal: 1100, tax: 134.56, total: 1234.56 }) as any).valid).toBe(true);
	});
	it('catches a mismatch', async () => {
		const r = await validateMath.run({ subtotal: 1100, tax: 134.56, total: 7234.56 }) as any;
		expect(r.valid).toBe(false);
		expect(r.delta).toBeCloseTo(-6000, 0);
	});
});

describe('validate_totals tool', () => {
	const items = [
		{ description: 'A', amount: '215.00', discount: '46.00', net_assessable_value: '169.00' },
		{ description: 'B', amount: '240.00', discount: '65.00', net_assessable_value: '175.00' },
		{ description: 'C', amount: '20.00', discount: '0.00', net_assessable_value: '20.00' },
	];
	it('passes the consistent Swiggy invoice (net items=subtotal, +tax=total, words=total)', async () => {
		const r = await validateTotals.run({
			line_items: items, subtotal: '364.00', tax: '18.20', total: '382.20',
			total_in_words: 'Three Hundred Eighty Two Rupees Twenty Paise Only',
		}) as any;
		expect(r.consistent).toBe(true);
		expect(r.issues).toHaveLength(0);
	});
	it('does NOT flag when only line-item sums differ (discounts) but totals reconcile', async () => {
		const gross = [
			{ description: 'A', quantity: 1, unit_price: 215, amount: 215 },
			{ description: 'B', quantity: 1, unit_price: 240, amount: 240 },
			{ description: 'C', quantity: 1, unit_price: 20, amount: 20 },
		]; // sum 475 ≠ net subtotal 364, but totals + words reconcile → consistent
		const r = await validateTotals.run({
			line_items: gross, subtotal: 364, tax: 18.2, total: 382.2,
			total_in_words: 'Three Hundred Eighty Two Rupees Twenty Paise Only',
		}) as any;
		expect(r.consistent).toBe(true);
		expect(r.soft_notes.length).toBeGreaterThan(0); // line-item diff noted, not decisive
	});
	it('does NOT flag a sub-rupee ROUND OFF gap (subtotal+tax off by 0.35, words match)', async () => {
		const r = await validateTotals.run({
			subtotal: 22686.99, tax: 4083.66, total: 26771.00,
			total_in_words: 'Twenty Six Thousand Seven Hundred Seventy One Only',
		}) as any;
		expect(r.consistent).toBe(true); // 0.35 round-off absorbed; words match exactly
	});
	it('accepts an explicit round_off for an exact match', async () => {
		const r = await validateTotals.run({ subtotal: 22686.99, tax: 4083.66, round_off: 0.35, total: 26771.00 }) as any;
		expect(r.consistent).toBe(true);
	});
	it('catches a mis-read total (words/math say 382.20, digits say 215.00)', async () => {
		const r = await validateTotals.run({
			subtotal: '364.00', tax: '18.20', total: '215.00',
			total_in_words: 'Three Hundred Eighty Two Rupees Twenty Paise Only',
		}) as any;
		expect(r.consistent).toBe(false);
		expect(r.issues.join(' ')).toContain('total');
	});
});

describe('coverageNote — catches incomplete extraction', () => {
	const text = '1 Widget A\n100.00\n2 Widget B\n200.00\n3 Widget C\n4 Widget D\n5 Widget E';
	it('flags when fewer line_items than rows in the document', () => {
		expect(coverageNote(text, { line_items: [{}, {}, {}] })).toContain('INCOMPLETE'); // 3 of 5
	});
	it('passes when every row is captured', () => {
		expect(coverageNote(text, { line_items: [{}, {}, {}, {}, {}] })).toBeNull();
	});
	it('no-op when there are no line_items to check', () => {
		expect(coverageNote(text, { vendor: 'X' })).toBeNull();
	});
});

describe('amount-in-words parsing', () => {
	it('wordsToNumber handles hundreds, scales (thousand/lakh)', () => {
		expect(wordsToNumber('three hundred eighty two')).toBe(382);
		expect(wordsToNumber('one thousand two hundred thirty four')).toBe(1234);
		expect(wordsToNumber('twelve lakh fifty thousand')).toBe(1250000);
	});
	it('parseAmountWords reads rupees + paise', () => {
		expect(parseAmountWords('Three Hundred Eighty Two Rupees Twenty Paise Only')).toBeCloseTo(382.20, 2);
		expect(parseAmountWords('One Thousand Rupees Only')).toBeCloseTo(1000, 2);
	});
});
