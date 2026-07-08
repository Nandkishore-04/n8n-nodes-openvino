/**
 * Prompt-based agent loop (ReAct-style). The model is told to reply with either a tool call
 * {"tool","args"} or a final answer {"final"}. We parse, dispatch, feed the result back, repeat.
 * No OVMS-native tool parser needed — portable across model servers.
 */
import { Tool, toolDescriptions } from './tools/builtIn';

export type ChatFn = (messages: Array<{ role: string; content: string }>) => Promise<string>;

export interface AgentResult {
	final: string;
	/** if the final answer is an object, its parsed form (so downstream can read fields) */
	finalData?: unknown;
	/** convenience: the routing decision, surfaced for a Switch node */
	decision?: string;
	/** why the agent reached this decision (for audit / metadata) */
	reason?: string;
	/** the agent's confidence in its decision, 0..1 (for metadata) */
	confidence?: number;
	/** the structured fields from the last extract_fields call (the real extracted data) */
	extracted?: unknown;
	iterations: Array<{ iter: number; tool?: string; args?: unknown; result?: unknown; raw: string }>;
	incomplete: boolean;
}

function stripThink(s: string): string {
	return s.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

// Pull the first balanced JSON object out of the model's text.
function extractJson(s: string): any | null {
	const start = s.indexOf('{');
	if (start === -1) return null;
	let depth = 0;
	for (let i = start; i < s.length; i++) {
		if (s[i] === '{') depth++;
		else if (s[i] === '}') {
			depth--;
			if (depth === 0) {
				try { return JSON.parse(s.slice(start, i + 1)); } catch { return null; }
			}
		}
	}
	return null;
}

// Detect runaway LLM degeneration (a repetition loop) so we flag for review instead of
// looping on — or returning — thousands of tokens of garbage. A valid tool call is never this big.
function looksDegenerate(s: string): boolean {
	if (s.length > 3000) return true;
	return /(\w{6,})\1{5,}/.test(s.slice(0, 2000)); // same chunk repeated back-to-back
}

function systemPrompt(userSystem: string, tools: Map<string, Tool>): string {
	return [
		userSystem.trim(),
		'',
		'You have these tools:',
		toolDescriptions(tools),
		'',
		'On each turn reply with ONLY one JSON object:',
		'  to call a tool:    {"tool": "<name>", "args": { ... }}',
		'  when finished:     {"final": "<your answer / decision>"}',
		'Do not add any text outside the JSON.',
	].join('\n');
}

export async function runAgent(opts: {
	chat: ChatFn;
	tools: Map<string, Tool>;
	system: string;
	input: string;
	maxIters?: number;
	noThink?: boolean;
}): Promise<AgentResult> {
	const { chat, tools, input } = opts;
	const maxIters = opts.maxIters ?? 8;
	const suffix = opts.noThink ? ' /no_think' : '';

	const messages: Array<{ role: string; content: string }> = [
		{ role: 'system', content: systemPrompt(opts.system, tools) },
		{ role: 'user', content: input + suffix },
	];
	const iterations: AgentResult['iterations'] = [];
	let extracted: unknown; // last extract_fields result — the real structured data
	let degenRetries = 0;   // small local models occasionally degenerate; retry once before flagging

	for (let i = 0; i < maxIters; i++) {
		const raw = stripThink(await chat(messages));

		// Runaway repetition (degenerate generation). A clean re-prompt usually clears a transient
		// glitch; only flag for review if it degenerates again (so a valid doc isn't falsely flagged).
		if (looksDegenerate(raw)) {
			if (degenRetries < 1) {
				degenRetries++;
				iterations.push({ iter: i, raw: raw.slice(0, 200) + '… [degenerate output — retrying]' });
				messages.push({ role: 'user', content: `Your last reply was malformed/repetitive. Reply again with ONLY a compact, valid JSON object.${suffix}` });
				continue;
			}
			const fd = { decision: 'flagged', reason: 'agent response repeatedly malformed (model degeneration) — needs human review', confidence: 0 };
			iterations.push({ iter: i, raw: raw.slice(0, 300) + '… [truncated degenerate output]' });
			return { final: JSON.stringify(fd), finalData: fd, decision: 'flagged', reason: fd.reason, confidence: 0, extracted, iterations, incomplete: false };
		}

		const parsed = extractJson(raw);
		messages.push({ role: 'assistant', content: raw });

		if (parsed && typeof parsed.final !== 'undefined') {
			iterations.push({ iter: i, raw });
			const fv = parsed.final;
			const isObj = fv !== null && typeof fv === 'object';
			return {
				final: isObj ? JSON.stringify(fv) : String(fv),
				finalData: isObj ? fv : undefined,
				decision: isObj ? (fv.decision as string | undefined) : undefined,
				reason: isObj ? (fv.reason as string | undefined) : undefined,
				confidence: isObj && fv.confidence != null ? Number(fv.confidence) : undefined,
				extracted,
				iterations,
				incomplete: false,
			};
		}

		if (parsed && parsed.tool) {
			const tool = tools.get(parsed.tool);
			const result = tool
				? await tool.run(parsed.args ?? {})
				: { error: `unknown tool '${parsed.tool}'` };
			if (parsed.tool === 'extract_fields') extracted = result;
			iterations.push({ iter: i, tool: parsed.tool, args: parsed.args, result, raw });
			messages.push({ role: 'user', content: `Tool result: ${JSON.stringify(result)}${suffix}` });
			continue;
		}

		// JSON failed to parse. A truncated FINAL — if we already have extracted data, salvage it:
		// pull the decision from the partial text and finish, rather than looping forever.
		if (/"final"\s*:/.test(raw)) {
			iterations.push({ iter: i, raw });
			const dec = raw.match(/"decision"\s*:\s*"(\w+)"/);
			const rsn = raw.match(/"reason"\s*:\s*"([^"]*)"/);
			if (extracted || dec) {
				return {
					final: dec ? dec[1] : 'enriched',
					decision: dec ? dec[1] : 'enriched',
					reason: rsn ? rsn[1] : undefined,
					extracted,
					iterations,
					incomplete: false,
				};
			}
		}
		// A truncated tool call → nudge a compact retry (don't return garbage as the answer).
		if (/"tool"\s*:/.test(raw) && i < maxIters - 1) {
			iterations.push({ iter: i, raw });
			messages.push({ role: 'user', content: `Your last reply was cut off. Reply with a COMPACT JSON object only.${suffix}` });
			continue;
		}

		// Otherwise it's a plain-text answer → treat as final.
		iterations.push({ iter: i, raw });
		return { final: raw, extracted, iterations, incomplete: false };
	}

	// Hit the iteration cap — return the best we have (with extracted data), flagged incomplete.
	return { final: iterations[iterations.length - 1]?.raw ?? '', extracted, iterations, incomplete: true };
}
