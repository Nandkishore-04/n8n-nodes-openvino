import { runAgent } from '../../../nodes/OpenVinoAgent/loop';
import { toolRegistry, validateMath, buildTools, ToolContext } from '../../../nodes/OpenVinoAgent/tools/builtIn';

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
});

describe('buildTools — all 7 tools', () => {
	const baseCtx = (over: Partial<ToolContext> = {}): ToolContext => ({
		chat: async () => '{"vendor":"Acme"}',
		http: async () => ({}),
		gatewayUrl: 'http://gateway:8000',
		noThink: true,
		...over,
	});

	it('registers all seven tools', () => {
		const tools = buildTools(baseCtx());
		expect([...tools.keys()].sort()).toEqual(
			['check_duplicate', 'extract_fields', 'flag_for_review', 'lookup_vendor', 'recall', 'retry_document_extraction', 'validate_math'],
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
