'use client';

import Link from 'next/link';
import { useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  ArrowRight,
  Check,
  ChevronDown,
  Cpu,
  FileText,
  Loader2,
  MessagesSquare,
  RefreshCw,
  ShieldCheck,
  X,
} from 'lucide-react';

interface Health {
  online: boolean;
  available_devices?: string[];
  asr?: boolean;
  tts?: boolean;
}

interface Probe {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
  fix?: string;
}

const TOOLS = [
  {
    href: '/documents',
    Icon: FileText,
    title: 'Document Processing',
    desc: 'Drop in PDFs and images — triaged, read, understood and stored automatically.',
    chips: ['CLIP · NPU', 'VLM OCR · GPU', 'Agent · GPU'],
  },
  {
    href: '/chat',
    Icon: MessagesSquare,
    title: 'Ask Your Documents',
    desc: 'Chat with everything you have processed. Grounded answers with cited sources.',
    chips: ['BGE · CPU', 'Hybrid search', 'Qwen3 · GPU'],
  },
];

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [stored, setStored] = useState<number | null>(null);
  const [probes, setProbes] = useState<Probe[] | null>(null);
  const [checking, setChecking] = useState(false);

  const runDiagnostics = useCallback(async () => {
    setChecking(true);
    try {
      const j = await (await fetch('/api/diagnostics', { cache: 'no-store' })).json();
      setProbes(j.probes ?? []);
    } catch {
      setProbes([{ id: 'app', label: 'Web app', ok: false, detail: 'Diagnostics route unreachable.' }]);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    fetch('/api/health', { cache: 'no-store' })
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ online: false }));
    fetch('/api/documents', { cache: 'no-store' })
      .then((r) => r.json())
      .then((j) => setStored((j.documents ?? []).length))
      .catch(() => setStored(null));
    runDiagnostics();
  }, [runDiagnostics]);

  const devices = health?.available_devices ?? [];

  return (
    <div className="flex h-full flex-col items-center justify-center overflow-y-auto px-6 py-10">
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="text-center"
      >
        <div className="mx-auto mb-5 flex w-fit items-center gap-2 rounded-full border border-brand-500/30 bg-brand-500/10 px-3.5 py-1.5 text-[11px] text-brand-300">
          <ShieldCheck size={12} />
          Private by design — nothing ever leaves this machine
        </div>
        <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl">
          Your documents. Your AI.
          <span className="block bg-gradient-to-r from-brand-300 via-brand-400 to-brand-500 bg-clip-text text-transparent">
            Your machine.
          </span>
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-sm leading-relaxed text-white/45">
          Two AI tools powered by n8n workflows and OpenVINO models, accelerated across the Intel
          NPU, GPU and CPU in this PC.
        </p>
      </motion.div>

      {/* Tool cards */}
      <div className="mt-10 grid w-full max-w-3xl gap-4 sm:grid-cols-2">
        {TOOLS.map(({ href, Icon, title, desc, chips }, i) => (
          <motion.div
            key={href}
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12 + i * 0.09, duration: 0.4, ease: 'easeOut' }}
            whileHover={{ y: -5 }}
          >
            <Link
              href={href}
              className="group flex h-full flex-col rounded-2xl border border-white/[0.08] bg-white/[0.02] p-5 transition-colors hover:border-brand-500/50 hover:bg-brand-500/[0.06] hover:shadow-[0_0_40px_rgba(108,36,240,0.15)]"
            >
              <div className="grid h-11 w-11 place-items-center rounded-xl bg-brand-500/15 text-brand-300 transition group-hover:bg-brand-500/25">
                <Icon size={20} />
              </div>
              <h2 className="mt-4 text-[15px] font-semibold text-white">{title}</h2>
              <p className="mt-1.5 flex-1 text-xs leading-relaxed text-white/40">{desc}</p>
              <div className="mt-4 flex flex-wrap gap-1.5">
                {chips.map((c) => (
                  <span
                    key={c}
                    className="rounded-md border border-white/[0.08] bg-white/[0.03] px-1.5 py-0.5 text-[10px] text-white/45"
                  >
                    {c}
                  </span>
                ))}
              </div>
              <span className="mt-4 flex items-center gap-1 text-xs font-medium text-brand-300 opacity-0 transition-opacity group-hover:opacity-100">
                Open <ArrowRight size={12} className="transition-transform group-hover:translate-x-0.5" />
              </span>
            </Link>
          </motion.div>
        ))}
      </div>

      {/* Live status */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5, duration: 0.4 }}
        className="mt-10 flex flex-wrap items-center justify-center gap-2 text-[11px]"
      >
        <span
          className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 ${
            health?.online
              ? 'border-brand-500/40 bg-brand-500/10 text-brand-300'
              : 'border-white/20 bg-white/[0.04] text-white/50'
          }`}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${health?.online ? 'animate-pulse bg-brand-400' : 'bg-white/40'}`} />
          {health === null ? 'Checking gateway…' : health.online ? 'AI gateway online' : 'Gateway offline — start it to begin'}
        </span>
        {devices.length > 0 && (
          <span className="flex items-center gap-1.5 rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 text-white/50">
            <Cpu size={11} className="text-brand-300" />
            {devices.join(' · ')}
          </span>
        )}
        {stored != null && (
          <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 text-white/50">
            {stored} document{stored === 1 ? '' : 's'} searchable
          </span>
        )}
      </motion.div>

      <ConnectionPanel probes={probes} checking={checking} onRecheck={runDiagnostics} />
    </div>
  );
}

// Every link to the backend, with the exact fix when one is down. Collapsed to a single
// green line when everything is connected, so it disappears once you're set up.
function ConnectionPanel({
  probes,
  checking,
  onRecheck,
}: {
  probes: Probe[] | null;
  checking: boolean;
  onRecheck: () => void;
}) {
  const allOk = probes != null && probes.every((p) => p.ok);
  const [open, setOpen] = useState(false);
  const broken = probes?.filter((p) => !p.ok) ?? [];
  // Auto-open the moment something is actually wrong — you shouldn't have to go looking.
  useEffect(() => {
    if (probes && !allOk) setOpen(true);
  }, [probes, allOk]);

  if (!probes) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.6 }}
      className="mt-6 w-full max-w-3xl"
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className={`flex w-full items-center gap-2.5 rounded-xl border px-4 py-2.5 text-left text-xs transition ${
          allOk
            ? 'border-brand-500/25 bg-brand-500/[0.06] text-brand-300 hover:bg-brand-500/10'
            : 'border-white/25 bg-white/[0.05] text-white/85 hover:bg-white/[0.08]'
        }`}
      >
        {allOk ? <Check size={13} /> : <X size={13} />}
        <span className="font-medium">
          {allOk
            ? 'All backend connections healthy'
            : `${broken.length} connection${broken.length === 1 ? '' : 's'} need attention`}
        </span>
        <ChevronDown size={13} className={`ml-auto transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="mt-2 animate-fade-in space-y-1.5 rounded-xl border border-white/[0.08] bg-white/[0.02] p-3">
          {probes.map((p) => (
            <div key={p.id} className="flex gap-2.5 rounded-lg px-2 py-1.5 text-xs">
              <span className={`mt-0.5 shrink-0 ${p.ok ? 'text-brand-300' : 'text-white/70'}`}>
                {p.ok ? <Check size={13} /> : <X size={13} />}
              </span>
              <div className="min-w-0 flex-1">
                <p className={p.ok ? 'text-white/70' : 'text-white/90'}>{p.label}</p>
                <p className="mt-0.5 break-words font-mono text-[10.5px] leading-relaxed text-white/35">{p.detail}</p>
                {!p.ok && p.fix && (
                  <p className="mt-1 rounded-md border border-brand-500/25 bg-brand-500/[0.07] px-2 py-1 text-[11px] text-brand-300">
                    Fix: {p.fix}
                  </p>
                )}
              </div>
            </div>
          ))}
          <button
            onClick={onRecheck}
            disabled={checking}
            className="mt-1 flex items-center gap-1.5 rounded-lg border border-white/[0.1] px-2.5 py-1.5 text-[11px] text-white/55 transition hover:text-white/90 disabled:opacity-40"
          >
            {checking ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
            Re-check
          </button>
        </div>
      )}
    </motion.div>
  );
}
