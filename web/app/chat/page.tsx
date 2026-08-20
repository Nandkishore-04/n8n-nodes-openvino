'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  Bot,
  ChevronDown,
  Cpu,
  FileText,
  Loader2,
  Search,
  Send,
  Sparkles,
  Trash2,
  User,
} from 'lucide-react';

interface Match {
  file: string;
  score: number | null;
  terms: number;
  via: string;
}
interface ChatMsg {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  sources?: string[];
  matches?: Match[];
  error?: boolean;
  pending?: boolean;
}

type Mode = 'rag' | 'agent';

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [storedCount, setStoredCount] = useState(0);
  const [latestDoc, setLatestDoc] = useState<string | undefined>();
  const [mode, setMode] = useState<Mode>('rag');
  // One session per mode+page-load: switching modes starts a clean thread, which avoids
  // the agent inheriting stale context from a previous line of questioning.
  const sessionId = useRef(`web-${Date.now().toString(36)}`);
  const scroller = useRef<HTMLDivElement>(null);

  // Light poll for the header count + suggestions (documents change rarely while chatting).
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const r = await fetch('/api/documents', { cache: 'no-store' });
        const j = await r.json();
        if (!alive) return;
        const docs = j.documents ?? [];
        setStoredCount(docs.length);
        setLatestDoc(docs[0]?.file_name);
      } catch {
        /* backend offline — chat input stays enabled; errors surface per-message */
      }
    };
    load();
    const t = setInterval(load, 10000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const ask = useCallback(
    async (q: string) => {
      const query = q.trim();
      if (!query || busy) return;
      setInput('');
      setBusy(true);
      const pendingId = `a-${Date.now()}`;
      setMessages((m) => [
        ...m,
        { id: `u-${Date.now()}`, role: 'user', text: query },
        { id: pendingId, role: 'assistant', text: '', pending: true },
      ]);

      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, mode, sessionId: sessionId.current }),
        });
        const json = await res.json();
        setMessages((m) =>
          m.map((msg) =>
            msg.id === pendingId
              ? json.error
                ? { ...msg, pending: false, error: true, text: json.error }
                : {
                    ...msg,
                    pending: false,
                    text: json.answer ?? 'No answer returned.',
                    sources: json.sources ?? [],
                    matches: json.matches ?? [],
                  }
              : msg,
          ),
        );
      } catch {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === pendingId
              ? { ...msg, pending: false, error: true, text: 'Request failed — check that n8n and the gateway are running.' }
              : msg,
          ),
        );
      } finally {
        setBusy(false);
      }
    },
    [busy, mode],
  );

  const suggestions = useMemo(() => {
    const s: string[] = [];
    if (latestDoc) s.push(`Summarize ${latestDoc}`);
    s.push('What taxes were charged on my invoices?', 'Which vendors appear in my documents?');
    return s.slice(0, 3);
  }, [latestDoc]);

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col px-6 py-6">
      {/* Page header */}
      <div className="flex shrink-0 items-center justify-between pb-4">
        <div className="flex items-center gap-2.5">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-brand-500/15 text-brand-300">
            <Sparkles size={16} />
          </span>
          <div>
            <h1 className="text-sm font-semibold text-white">Ask your documents</h1>
            <p className="text-[11px] text-white/40">
              {mode === 'rag'
                ? `Hybrid search over ${storedCount} stored document${storedCount === 1 ? '' : 's'} · grounded answers with sources`
                : 'Agent decides: search documents, query the database, or answer directly'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Two answering paths, side by side: the deterministic retrieval pipeline and
              the agent that routes between tools. Switching starts a fresh thread. */}
          <div className="flex rounded-lg border border-white/[0.08] bg-white/[0.03] p-0.5 text-[11px]">
            {(['rag', 'agent'] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => {
                  if (m === mode) return;
                  setMode(m);
                  setMessages([]);
                  sessionId.current = `web-${Date.now().toString(36)}`;
                }}
                title={
                  m === 'rag'
                    ? 'Straight to the RAG pipeline — always retrieves, always cites'
                    : 'The agentic chatbot — picks its own tool per question'
                }
                className={`rounded-md px-2.5 py-1 transition ${
                  mode === m ? 'bg-brand-500/25 text-brand-200 ring-1 ring-inset ring-brand-400/40' : 'text-white/45 hover:text-white/80'
                }`}
              >
                {m === 'rag' ? 'Direct RAG' : 'Agent'}
              </button>
            ))}
          </div>
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              className="flex items-center gap-1.5 rounded-lg border border-white/[0.08] px-2.5 py-1.5 text-[11px] text-white/50 transition hover:border-white/20 hover:text-white/90"
            >
              <Trash2 size={12} />
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div ref={scroller} className="glass min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="grid h-14 w-14 place-items-center rounded-2xl bg-brand-500/10 text-brand-300">
              <Search size={24} />
            </div>
            <p className="mt-4 text-sm font-medium text-white/80">
              {storedCount > 0 ? 'Ask anything about your processed documents' : 'Process a document first — then ask about it here'}
            </p>
            <p className="mt-1 max-w-sm text-xs leading-relaxed text-white/35">
              Your question is embedded on the CPU, matched against the vector store (meaning + keywords), and answered by
              Qwen3 on the GPU — with the source files cited.
            </p>
            {storedCount > 0 && (
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {suggestions.map((s) => (
                  <button
                    key={s}
                    onClick={() => ask(s)}
                    className="max-w-[16rem] truncate rounded-full border border-brand-500/30 bg-brand-500/[0.08] px-3.5 py-1.5 text-xs text-brand-300 transition hover:border-brand-400/60 hover:bg-brand-500/15"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {messages.map((m) => (
          <MessageBubble key={m.id} msg={m} />
        ))}
      </div>

      {/* Input */}
      <div className="shrink-0 pt-3">
        <div className="flex items-end gap-2.5">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                ask(input);
              }
            }}
            rows={1}
            placeholder="Ask about your documents…  (Enter to send)"
            className="max-h-32 min-h-[2.75rem] flex-1 resize-none rounded-xl border border-white/[0.1] bg-white/[0.03] px-4 py-2.5 text-sm text-white placeholder:text-white/25 focus:border-brand-500/60 focus:outline-none focus:ring-2 focus:ring-brand-500/25"
          />
          <button
            onClick={() => ask(input)}
            disabled={busy || !input.trim()}
            className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-600 text-white shadow-lg shadow-brand-500/25 transition hover:from-brand-400 hover:to-brand-500 disabled:opacity-30 disabled:shadow-none"
            title="Send"
          >
            {busy ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Message bubble ────────────────────────────────────────────────────────────
function MessageBubble({ msg }: { msg: ChatMsg }) {
  if (msg.role === 'user') {
    return (
      <div className="flex animate-fade-up justify-end gap-2.5">
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-gradient-to-br from-brand-500 to-brand-600 px-4 py-2.5 text-sm text-white shadow-md shadow-brand-500/15">
          {msg.text}
        </div>
        <span className="mt-1 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-white/[0.08] text-white/60">
          <User size={13} />
        </span>
      </div>
    );
  }

  return (
    <div className="flex animate-fade-up gap-2.5">
      <span
        className={`mt-1 grid h-7 w-7 shrink-0 place-items-center rounded-full ${
          msg.error ? 'bg-white/[0.1] text-white/70' : 'bg-brand-500/15 text-brand-300'
        }`}
      >
        <Bot size={13} />
      </span>
      <div className="min-w-0 max-w-[85%]">
        {msg.pending ? (
          <Thinking />
        ) : (
          <div
            className={`rounded-2xl rounded-tl-md border px-4 py-3 text-sm leading-relaxed ${
              msg.error ? 'border-white/25 bg-white/[0.05] text-white/85' : 'border-white/[0.08] bg-white/[0.03] text-white/85'
            }`}
          >
            {msg.error && (
              <p className="mb-1 flex items-center gap-1.5 text-[11px] font-medium text-white/60">
                <AlertCircle size={12} /> Something needs attention
              </p>
            )}
            <p className="whitespace-pre-wrap">{msg.text}</p>

            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5 border-t border-white/[0.08] pt-2.5">
                {msg.sources.map((s) => (
                  <span
                    key={s}
                    className="flex items-center gap-1.5 rounded-full border border-brand-500/35 bg-brand-500/10 px-2.5 py-1 text-[11px] text-brand-300"
                    title={s}
                  >
                    <FileText size={11} />
                    <span className="max-w-[12rem] truncate">{s}</span>
                  </span>
                ))}
              </div>
            )}

            {msg.matches && msg.matches.length > 0 && <RetrievalDetails matches={msg.matches} />}
          </div>
        )}
      </div>
    </div>
  );
}

// Staged captions while the answer is produced — mirrors the real pipeline steps.
function Thinking() {
  const steps = [
    { Icon: Cpu, text: 'Embedding your question (BGE · CPU)' },
    { Icon: Search, text: 'Searching the vector store — meaning + keywords' },
    { Icon: Sparkles, text: 'Writing a grounded answer (Qwen3 · GPU)' },
  ];
  const [i, setI] = useState(0);
  useEffect(() => {
    const t1 = setTimeout(() => setI(1), 1600);
    const t2 = setTimeout(() => setI(2), 4200);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, []);
  const { Icon, text } = steps[i];
  return (
    <div className="flex items-center gap-3 rounded-2xl rounded-tl-md border border-white/[0.08] bg-white/[0.03] px-4 py-3">
      <span className="flex gap-1">
        <span className="h-1.5 w-1.5 animate-blink rounded-full bg-brand-400" />
        <span className="h-1.5 w-1.5 animate-blink rounded-full bg-brand-400 [animation-delay:0.2s]" />
        <span className="h-1.5 w-1.5 animate-blink rounded-full bg-brand-400 [animation-delay:0.4s]" />
      </span>
      <span key={i} className="flex animate-fade-in items-center gap-2 text-xs text-white/50">
        <Icon size={13} className="text-brand-300" />
        {text}…
      </span>
    </div>
  );
}

// Collapsible "how this answer was retrieved" — chunk scores + which search arm found each.
function RetrievalDetails({ matches }: { matches: Match[] }) {
  const [open, setOpen] = useState(false);
  const viaCls: Record<string, string> = {
    dense: 'border-white/20 text-white/60',
    keyword: 'border-brand-500/30 bg-brand-500/[0.07] text-brand-300/90',
    'dense+keyword': 'border-brand-400/60 bg-brand-500/15 text-brand-300',
  };
  return (
    <div className="mt-2.5">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-[11px] text-white/40 transition hover:text-white/80"
      >
        <ChevronDown size={12} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
        {open ? 'Hide' : 'Show'} retrieval details · {matches.length} chunk{matches.length === 1 ? '' : 's'}
      </button>
      {open && (
        <div className="mt-2 animate-fade-in space-y-1">
          {matches.map((m, idx) => (
            <div key={idx} className="flex items-center gap-2 rounded-lg bg-white/[0.03] px-2.5 py-1.5 text-[11px] text-white/50">
              <span className="max-w-[11rem] truncate text-white/75" title={m.file}>
                {m.file}
              </span>
              <span className={`rounded-full border px-1.5 py-px text-[10px] ${viaCls[m.via] ?? 'border-white/15'}`}>{m.via}</span>
              {m.score != null && <span className="ml-auto tabular-nums">sim {Number(m.score).toFixed(2)}</span>}
              {m.terms > 0 && (
                <span className="tabular-nums text-white/35">
                  · {m.terms} term{m.terms > 1 ? 's' : ''}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
