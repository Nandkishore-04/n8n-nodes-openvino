/**
 * Agent tools. Each tool has a description (shown to the LLM) and a run(args) impl.
 *
 * Pure tools (validate_math, flag_for_review) need no context. The rest need services
 * (LLM, gateway, Qdrant, external workflows) injected via ToolContext, so they're created
 * by buildTools(ctx). Tools whose backend isn't configured degrade gracefully — they return
 * a clear note instead of throwing, so the agent can reason about the gap.
 */

export interface Tool {
	name: string;
	description: string;
	run: (args: Record<string, unknown>) => Promise<unknown> | unknown;
}

export type ChatFn = (messages: Array<{ role: string; content: string }>) => Promise<string>;
export type HttpFn = (opts: Record<string, unknown>) => Promise<any>;

export interface ToolContext {
	chat: ChatFn;
	http: HttpFn;
	gatewayUrl: string;
	apiKey?: string;
	noThink: boolean;
	/** the document text being analyzed — so tools needn't have the model re-emit it */
	documentText?: string;
	/** base64 + filename of the source document, enabling retry_document_extraction */
	sourceFile?: { dataB64: string; filename: string };
	/** Workflow-as-Tool / HTTP endpoints */
	checkDuplicateUrl?: string;
	lookupVendorUrl?: string;
	/** memory */
	qdrant?: { url: string; collection: string };
	embeddingsModel?: string;
	/** curated reference knowledge base (separate Qdrant collection) for knowledge_search */
	knowledgeCollection?: string;
}

function stripThink(s: string): string {
	return s.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

function extractJson(s: string): any | null {
	const start = s.indexOf('{');
	if (start === -1) return null;
	let depth = 0;
	for (let i = start; i < s.length; i++) {
		if (s[i] === '{') depth++;
		else if (s[i] === '}' && --depth === 0) {
			try { return JSON.parse(s.slice(start, i + 1)); } catch { return null; }
		}
	}
	return null;
}

// ── Pure tools (no context) ─────────────────────────────────────────────────

export const validateMath: Tool = {
	name: 'validate_math',
	description: 'validate_math(subtotal, tax, total) → checks subtotal + tax == total',
	run: (args) => {
		const subtotal = Number(args.subtotal ?? 0);
		const tax = Number(args.tax ?? 0);
		const total = Number(args.total ?? 0);
		const expected = Math.round((subtotal + tax) * 100) / 100;
		const delta = Math.round((expected - total) * 100) / 100;
		return { valid: Math.abs(delta) < 0.01, expected_total: expected, actual_total: total, delta };
	},
};

export const flagForReview: Tool = {
	name: 'flag_for_review',
	description: 'flag_for_review(reason, severity) → marks the document for human review',
	run: (args) => ({
		flagged: true,
		reason: String(args.reason ?? 'unspecified'),
		severity: String(args.severity ?? 'medium'),
	}),
};

// ── Amount-in-words → number (catches a confidently mis-read total) ─────────────
const NUM_WORDS: Record<string, number> = {
	zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9,
	ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16,
	seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20, thirty: 30, forty: 40, fifty: 50,
	sixty: 60, seventy: 70, eighty: 80, ninety: 90,
};
const NUM_SCALES: Record<string, number> = {
	thousand: 1000, lakh: 100000, lakhs: 100000, crore: 10000000, crores: 10000000,
	million: 1000000, billion: 1000000000,
};

export function wordsToNumber(text: string): number | null {
	const words = text.toLowerCase().replace(/[^a-z\s]/g, ' ').split(/\s+/).filter(Boolean);
	let total = 0, current = 0, seen = false;
	for (const w of words) {
		if (w in NUM_WORDS) { current += NUM_WORDS[w]; seen = true; }
		else if (w === 'hundred') { current = (current || 1) * 100; seen = true; }
		else if (w in NUM_SCALES) { total += (current || 1) * NUM_SCALES[w]; current = 0; seen = true; }
		// 'and', 'only', 'rupees', 'paise', etc. are ignored
	}
	return seen ? total + current : null;
}

// "Three Hundred Eighty Two Rupees Twenty Paise Only" → 382.20
export function parseAmountWords(text: string): number | null {
	const t = text.toLowerCase();
	const curRe = /rupees?|dollars?|\brs\b/;
	let rupeesText = t, paiseText = '';
	const paiseIdx = t.indexOf('paise');
	if (paiseIdx >= 0) {
		const before = t.slice(0, paiseIdx);
		const ri = before.search(curRe);
		if (ri >= 0) { rupeesText = before.slice(0, ri); paiseText = before.slice(ri).replace(curRe, ''); }
		else { rupeesText = ''; paiseText = before; }
	} else {
		const ri = t.search(curRe);
		if (ri >= 0) rupeesText = t.slice(0, ri);
	}
	const rupees = wordsToNumber(rupeesText);
	const paise = paiseText ? wordsToNumber(paiseText) : 0;
	if (rupees === null && !paise) return null;
	return Math.round(((rupees ?? 0) + (paise ?? 0) / 100) * 100) / 100;
}

// validate_totals — deterministic consistency checks across the extracted financial fields.
// This is the net that catches a confidently-wrong number (the VLM's confidence won't).
export const validateTotals: Tool = {
	name: 'validate_totals',
	description: 'validate_totals(line_items, subtotal, tax, total, total_in_words) → deterministic checks: line items sum to subtotal, subtotal+tax=total, and the amount-in-words matches the digit total. Call after extracting financials; if not consistent, the value was likely mis-read — flag_for_review.',
	run: (args) => {
		const num = (x: unknown) => {
			const n = Number(String(x ?? '').replace(/[^0-9.\-]/g, ''));
			return Number.isFinite(n) ? n : NaN;
		};
		const r2 = (n: number) => Math.round(n * 100) / 100;
		const close = (a: number, b: number) => Math.abs(a - b) < 0.02;
		type Chk = { name: string; ok: boolean; expected: number; actual: number };
		const strong: Chk[] = []; // authoritative — a failure here means a number was mis-read
		const soft: Chk[] = [];   // informational — fails routinely on per-line discounts/taxes

		const items = Array.isArray(args.line_items) ? (args.line_items as Array<Record<string, unknown>>) : [];
		const subtotal = num(args.subtotal);
		const tax = num(args.tax ?? args.tax_total ?? 0);
		const total = num(args.total);

		if (!Number.isNaN(subtotal) && !Number.isNaN(total)) {
			const hasRoundOff = args.round_off !== undefined && args.round_off !== null && String(args.round_off).trim() !== '';
			const roundOff = hasRoundOff ? num(args.round_off) : 0;
			const exp = r2(subtotal + (Number.isNaN(tax) ? 0 : tax) + roundOff);
			// invoices round the grand total to the nearest unit; absorb up to ~1 when no round_off was given
			const tol = hasRoundOff ? 0.05 : 1.05;
			strong.push({ name: 'subtotal+tax=total', ok: Math.abs(exp - total) <= tol, expected: exp, actual: total });
		}
		if (args.total_in_words && !Number.isNaN(total)) {
			const w = parseAmountWords(String(args.total_in_words));
			if (w !== null) strong.push({ name: 'amount_in_words=total', ok: close(w, total), expected: w, actual: total });
		}
		if (items.length && !Number.isNaN(subtotal)) {
			// gross vs net vs after-discount — accept any plausible interpretation, tolerate field-name variety
			const amt = (it: Record<string, unknown>) => num(it.amount ?? it.amount_rs ?? it.price_before_tax ?? it.price ?? it.total ?? it.value);
			const net = (it: Record<string, unknown>) => num(it.net_assessable_value ?? it.net_assessable_value_rs ?? it.amount ?? it.amount_rs ?? it.price_before_tax);
			const sumAmt = r2(items.reduce((s, it) => s + amt(it), 0));
			const sumNet = r2(items.reduce((s, it) => s + net(it), 0));
			const sumDisc = r2(items.reduce((s, it) => s + (amt(it) - num(it.discount ?? 0)), 0));
			const ok = [sumAmt, sumNet, sumDisc].some((v) => close(v, subtotal));
			soft.push({ name: 'line_items_sum≈subtotal', ok, expected: subtotal, actual: sumNet });
		}

		// Consistency rides on the STRONG checks (totals math + amount-in-words). Line-item sums
		// differ routinely from per-line discounts/taxes, so they only decide when nothing stronger ran.
		const deciding = strong.length ? strong : soft;
		const issues = deciding.filter((c) => !c.ok).map((c) => `${c.name}: expected ${c.expected}, got ${c.actual}`);
		const softNotes = soft.filter((c) => !c.ok).map((c) => `${c.name}: ${c.actual} vs ${c.expected} (likely discounts — not decisive)`);
		return { consistent: issues.length === 0, checks: [...strong, ...soft], issues, soft_notes: softNotes };
	},
};

// The LLM sometimes pads/mangles JSON KEYS (e.g. " subtotal ") — normalize keys so the stored
// record is clean and queryable. Trims leading/trailing whitespace recursively; values untouched.
function trimKeys(v: unknown): unknown {
	if (Array.isArray(v)) return v.map(trimKeys);
	if (v && typeof v === 'object') {
		const out: Record<string, unknown> = {};
		for (const [k, val] of Object.entries(v as Record<string, unknown>)) out[k.trim()] = trimKeys(val);
		return out;
	}
	return v;
}

// Coverage check: did the extraction capture every line-item row the document actually has?
// Counts numbered rows in the OCR text ("1 Desc", "2. …", "14 …") vs the extracted line_items —
// so a truncated/summarized extraction (e.g. 3 of 14 rows) is caught instead of shipping silently.
export function coverageNote(text: string, fields: unknown): string | null {
	const f = fields as Record<string, unknown> | null;
	const li = f && Array.isArray(f.line_items) ? f.line_items.length : null;
	if (li === null) return null;
	const rows = (text.match(/^\s*\d{1,3}[\s.)]+\D/gm) ?? []).length;
	if (rows >= 2 && li < rows) {
		return `document has ~${rows} line-item rows but only ${li} were captured — extraction likely INCOMPLETE; re-extract ALL rows or flag_for_review`;
	}
	return null;
}

// ── Context-dependent tools ──────────────────────────────────────────────────

// extract_fields — re-invokes the LLM with a caller-chosen schema, returns structured JSON.
// Self-reports a _coverage_warning when it looks like rows were dropped.
function extractFields(ctx: ToolContext): Tool {
	return {
		name: 'extract_fields',
		description: 'extract_fields(schema) → fills your schema from the document (do NOT pass the text; it is already loaded). Capture EVERY row — if it returns _coverage_warning, rows were dropped.',
		run: async (args) => {
			const schema = args.schema ?? {};
			// Prefer the injected document text; the model should NOT re-emit it (truncates tool calls).
			const text = String(args.text ?? ctx.documentText ?? '');
			const suffix = ctx.noThink ? ' /no_think' : '';
			const content = await ctx.chat([
				{ role: 'system', content: `Extract these fields from the document and reply ONLY with JSON matching this schema: ${JSON.stringify(schema)}. Capture EVERY row/line item — never summarize, omit, or stop early. Copy every text value EXACTLY as written (names, descriptions, codes, GSTINs) — do NOT fix spelling, change case, expand abbreviations, or paraphrase; transcribe verbatim.` },
				{ role: 'user', content: `${text}${suffix}` },
			]);
			const parsed = (trimKeys(extractJson(stripThink(content))) as Record<string, unknown>) ?? { _raw: stripThink(content) };
			const cov = coverageNote(ctx.documentText ?? '', parsed);
			if (cov) (parsed as Record<string, unknown>)._coverage_warning = cov;
			return parsed;
		},
	};
}

// retry_document_extraction — re-runs OCR on the source doc with enhancement / higher DPI.
function retryDocumentExtraction(ctx: ToolContext): Tool {
	return {
		name: 'retry_document_extraction',
		description: 'retry_document_extraction(enhance, dpi) → re-runs OCR on the source document (use when text is poor)',
		run: async (args) => {
			if (!ctx.sourceFile) return { error: 'no source document available to retry (agent received text only)' };
			const dpi = Number(args.dpi ?? 300);
			const res = await ctx.http({
				method: 'POST',
				url: `${ctx.gatewayUrl}/v1/document/infer`,
				headers: { 'Content-Type': 'application/json' },
				body: { data: ctx.sourceFile.dataB64, filename: ctx.sourceFile.filename, dpi, enhance: Boolean(args.enhance ?? true) },
				json: true,
			});
			return { text: res.text, confidence: res.confidence, source: res.source, attempt: 'retry' };
		},
	};
}

// check_duplicate — Workflow-as-Tool: POST to a configured n8n webhook / service.
function checkDuplicate(ctx: ToolContext): Tool {
	return {
		name: 'check_duplicate',
		description: 'check_duplicate(document_id) → has this document been processed before?',
		run: async (args) => {
			if (!ctx.checkDuplicateUrl) return { exists: false, note: 'no dedup backend configured' };
			return ctx.http({ method: 'POST', url: ctx.checkDuplicateUrl, body: { document_id: args.document_id }, json: true });
		},
	};
}

// lookup_vendor — Workflow-as-Tool / HTTP: enrich a vendor by name.
function lookupVendor(ctx: ToolContext): Tool {
	return {
		name: 'lookup_vendor',
		description: 'lookup_vendor(name) → vendor details (id, tax_id, payment terms)',
		run: async (args) => {
			if (!ctx.lookupVendorUrl) return { found: false, note: 'no vendor backend configured' };
			return ctx.http({ method: 'POST', url: ctx.lookupVendorUrl, body: { name: args.name }, json: true });
		},
	};
}

// recall — semantic memory: embed the query (BGE via gateway) → search Qdrant.
function recall(ctx: ToolContext): Tool {
	return {
		name: 'recall',
		description: 'recall(query, top_k) → similar past documents from memory',
		run: async (args) => {
			if (!ctx.qdrant) return { hits: [], note: 'recall unavailable (no memory configured)' };
			try {
				const emb = await ctx.http({
					method: 'POST',
					url: `${ctx.gatewayUrl}/v1/embeddings`,
					body: { model: ctx.embeddingsModel ?? 'bge-small-en', input: String(args.query ?? '') },
					json: true,
				});
				const vector = emb?.data?.[0]?.embedding;
				if (!vector) return { hits: [], note: 'embedding unavailable (BGE model not loaded)' };
				const search = await ctx.http({
					method: 'POST',
					url: `${ctx.qdrant.url}/collections/${ctx.qdrant.collection}/points/search`,
					body: { vector, limit: Number(args.top_k ?? 3), with_payload: true },
					json: true,
				});
				return { hits: search?.result ?? [] };
			} catch (e) {
				return { hits: [], note: `recall failed: ${(e as Error).message}` };
			}
		},
	};
}

// ── Registry ─────────────────────────────────────────────────────────────────

// knowledge_search — curated reference KB (playbooks / business rules / domain facts).
// Same BGE→Qdrant path as recall, but a separate admin-seeded collection. Fully local.
function knowledgeSearch(ctx: ToolContext): Tool {
	return {
		name: 'knowledge_search',
		description: 'knowledge_search(query, top_k) → reference knowledge for this document type (extraction playbook, business rules, domain facts). Call BEFORE extracting to learn how to handle the document.',
		run: async (args) => {
			if (!ctx.qdrant) return { hits: [], note: 'knowledge base unavailable (no Qdrant configured)' };
			const collection = ctx.knowledgeCollection || 'knowledge_base';
			try {
				const emb = await ctx.http({
					method: 'POST',
					url: `${ctx.gatewayUrl}/v1/embeddings`,
					body: { model: ctx.embeddingsModel ?? 'bge-small-en', input: String(args.query ?? '') },
					json: true,
				});
				const vector = emb?.data?.[0]?.embedding;
				if (!vector) return { hits: [], note: 'embedding unavailable (BGE model not loaded)' };
				const search = await ctx.http({
					method: 'POST',
					url: `${ctx.qdrant.url}/collections/${collection}/points/search`,
					body: { vector, limit: Number(args.top_k ?? 3), with_payload: true },
					json: true,
				});
				return { hits: (search?.result ?? []).map((h: any) => ({ text: h?.payload?.text, topic: h?.payload?.topic, score: h?.score })) };
			} catch (e) {
				return { hits: [], note: `knowledge_search failed: ${(e as Error).message}` };
			}
		},
	};
}

export function buildTools(ctx: ToolContext): Map<string, Tool> {
	const all: Tool[] = [
		validateMath,
		validateTotals,
		flagForReview,
		extractFields(ctx),
		retryDocumentExtraction(ctx),
		checkDuplicate(ctx),
		lookupVendor(ctx),
		recall(ctx),
		knowledgeSearch(ctx),
	];
	const map = new Map<string, Tool>();
	for (const t of all) map.set(t.name, t);
	return map;
}

// Back-compat: a no-context registry with just the pure tools (used by unit tests).
export function toolRegistry(extra: Tool[] = []): Map<string, Tool> {
	const map = new Map<string, Tool>();
	for (const t of [validateMath, validateTotals, flagForReview, ...extra]) map.set(t.name, t);
	return map;
}

export function toolDescriptions(tools: Map<string, Tool>): string {
	return [...tools.values()].map((t) => `- ${t.description}`).join('\n');
}
