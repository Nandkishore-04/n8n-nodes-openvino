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

// ── Context-dependent tools ──────────────────────────────────────────────────

// extract_fields — re-invokes the LLM with a caller-chosen schema, returns structured JSON.
function extractFields(ctx: ToolContext): Tool {
	return {
		name: 'extract_fields',
		description: 'extract_fields(schema) → fills your schema from the document (do NOT pass the text; it is already loaded)',
		run: async (args) => {
			const schema = args.schema ?? {};
			// Prefer the injected document text; the model should NOT re-emit it (truncates tool calls).
			const text = String(args.text ?? ctx.documentText ?? '');
			const suffix = ctx.noThink ? ' /no_think' : '';
			const content = await ctx.chat([
				{ role: 'system', content: `Extract these fields from the document and reply ONLY with JSON matching this schema: ${JSON.stringify(schema)}` },
				{ role: 'user', content: `${text}${suffix}` },
			]);
			return extractJson(stripThink(content)) ?? { _raw: stripThink(content) };
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

export function buildTools(ctx: ToolContext): Map<string, Tool> {
	const all: Tool[] = [
		validateMath,
		flagForReview,
		extractFields(ctx),
		retryDocumentExtraction(ctx),
		checkDuplicate(ctx),
		lookupVendor(ctx),
		recall(ctx),
	];
	const map = new Map<string, Tool>();
	for (const t of all) map.set(t.name, t);
	return map;
}

// Back-compat: a no-context registry with just the pure tools (used by unit tests).
export function toolRegistry(extra: Tool[] = []): Map<string, Tool> {
	const map = new Map<string, Tool>();
	for (const t of [validateMath, flagForReview, ...extra]) map.set(t.name, t);
	return map;
}

export function toolDescriptions(tools: Map<string, Tool>): string {
	return [...tools.values()].map((t) => `- ${t.description}`).join('\n');
}
