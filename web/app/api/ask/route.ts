import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

// Two answering paths, both proxied server-side so the browser never needs CORS to n8n:
//
//  mode 'rag'   -> WF2's webhook directly. Deterministic: always retrieves, always
//                  returns {answer, sources, matches}. This is the reliable demo path.
//  mode 'agent' -> the agentic chatbot, which decides between searching documents,
//                  running a SQL query over the metadata, or answering directly.
//                  Richer, but it is an LLM making routing decisions on a local 8B.
const RAG_WEBHOOK_URL = process.env.N8N_RAG_WEBHOOK_URL || 'http://localhost:5678/webhook/rag-query';
const AGENT_WEBHOOK_URL = process.env.N8N_AGENT_WEBHOOK_URL || 'http://localhost:5678/webhook/agent-chat';

// The answer path is embed → hybrid search → Qwen3 on the GPU, and the gateway runs one
// inference at a time, so a question queued behind an OCR job can legitimately take a while.
const TIMEOUT_MS = 180_000;

/** The agent replies in prose and appends "Sources: a.pdf, b.pdf" (its prompt tells it to).
 *  Lift that line out so the UI can render source chips exactly like the RAG path. */
function splitSources(text: string): { answer: string; sources: string[] } {
  const m = text.match(/\n\s*sources?\s*:\s*(.+)\s*$/i);
  if (!m) return { answer: text.trim(), sources: [] };
  const sources = m[1]
    .split(/[,;]/)
    .map((s) => s.trim().replace(/^["'[]+|["'\]]+$/g, ''))
    .filter(Boolean);
  return { answer: text.slice(0, m.index).trim(), sources };
}

export async function POST(req: NextRequest) {
  let query = '';
  let mode: 'rag' | 'agent' = 'rag';
  let sessionId = 'web';
  try {
    const body = await req.json();
    query = String(body?.query ?? '').trim();
    if (body?.mode === 'agent') mode = 'agent';
    if (body?.sessionId) sessionId = String(body.sessionId);
  } catch {
    /* fall through to the empty-query check */
  }
  if (!query) {
    return NextResponse.json({ error: 'Ask a question first.' }, { status: 400 });
  }

  const url = mode === 'agent' ? AGENT_WEBHOOK_URL : RAG_WEBHOOK_URL;
  const payload = mode === 'agent' ? { chatInput: query, sessionId } : { query };

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });

    if (res.status === 404) {
      return NextResponse.json(
        {
          error:
            mode === 'agent'
              ? 'The agent workflow is not listening on /webhook/agent-chat. Add a Webhook trigger to it (see setup notes) and switch the workflow to Active.'
              : 'The RAG workflow is not listening. In n8n, open "RAG Q&A (WF2)" and switch it to Active.',
        },
        { status: 502 },
      );
    }
    if (!res.ok) {
      return NextResponse.json(
        { error: `n8n answered with HTTP ${res.status}. Check that workflow's execution log for the failing node.` },
        { status: 502 },
      );
    }

    const data = await res.json();

    if (mode === 'rag') {
      // WF2 already returns the exact shape the UI wants.
      return NextResponse.json({ ...data, mode });
    }

    // The agent's shape varies with how the workflow ends (Respond node, agent output,
    // or a single-item array), so normalize whatever came back into {answer, sources}.
    const raw = Array.isArray(data) ? data[0] ?? {} : data ?? {};
    const text = String(raw.output ?? raw.answer ?? raw.text ?? raw.message ?? '').trim();
    if (!text) {
      return NextResponse.json(
        { error: 'The agent returned an empty reply — check its execution log (it may have hit max iterations).', mode },
        { status: 502 },
      );
    }
    const { answer, sources } = splitSources(text);
    return NextResponse.json({ answer, sources, matches: [], mode });
  } catch (e) {
    const err = e as Error;
    const timedOut = err.name === 'TimeoutError' || err.name === 'AbortError';
    return NextResponse.json(
      {
        error: timedOut
          ? 'That took too long. The gateway may be busy processing documents — try again in a moment.'
          : 'Could not reach n8n. Is it running on :5678?',
        mode,
      },
      { status: 502 },
    );
  }
}
