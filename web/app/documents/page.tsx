'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { relTime } from '@/lib/format';
import {
  AlertCircle,
  Bell,
  BellOff,
  Check,
  ChevronDown,
  Clock,
  Copy,
  FileText,
  Loader2,
  Upload,
  X,
} from 'lucide-react';

// Purple "OV" badge for OS notifications (inline data URI — nothing loads over the network).
const NOTIF_ICON =
  'data:image/svg+xml,' +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#6c24f0"/><text x="32" y="43" font-family="Arial,sans-serif" font-size="30" font-weight="bold" fill="#fff" text-anchor="middle">OV</text></svg>',
  );

// ── Types ─────────────────────────────────────────────────────────────────────
interface FileEntry {
  name: string;
  size: number;
  mtime: number;
}
type FolderData = Record<'incoming' | 'processing' | 'processed' | 'failed' | 'rejected', FileEntry[]>;

interface DocumentRow {
  id: number;
  file_name: string;
  document_type: string | null;
  decision: string | null;
  reason: string | null;
  confidence: number | null;
  ocr_confidence: number | null;
  fields: Record<string, unknown> | null;
  summary: string | null;
  status: string;
  created_at: string;
}

type Stage = 'queued' | 'analyzing' | 'stalled' | 'stored' | 'flagged' | 'duplicate' | 'rejected' | 'failed';

// A file sits in processing/ only while WF1 is actively working on it. If a run crashed
// or n8n restarted mid-document, the claim is never released and the file stays there
// forever — so anything older than this is stuck, not busy, and we say so.
const STALL_AFTER_MS = 10 * 60 * 1000;
interface JourneyItem {
  name: string;
  stage: Stage;
  time: number;
  size?: number;
  doc?: DocumentRow;
}

const POLL_MS = 2000;

const STAGE: Record<
  Stage,
  { label: string; hint: string; cls: string; dot: string; Icon: typeof Check; active?: boolean }
> = {
  queued: {
    label: 'Queued',
    hint: 'Waiting for the pipeline to claim it',
    cls: 'border-white/15 bg-white/[0.04] text-white/60',
    dot: 'bg-white/40',
    Icon: Clock,
  },
  analyzing: {
    label: 'Analyzing',
    hint: 'CLIP triage (NPU) → OCR (GPU) → agent',
    cls: 'border-brand-400/50 bg-brand-500/15 text-brand-300',
    dot: 'bg-brand-400',
    Icon: Loader2,
    active: true,
  },
  stalled: {
    label: 'Stalled',
    hint: 'Claimed but never finished — move it back to incoming/ to retry',
    cls: 'border-white/30 bg-white/[0.06] text-white/85',
    dot: 'bg-white/70',
    Icon: AlertCircle,
  },
  stored: {
    label: 'Stored',
    hint: 'Extracted, validated & searchable',
    cls: 'border-brand-500/40 bg-brand-500/10 text-brand-300',
    dot: 'bg-brand-500',
    Icon: Check,
  },
  flagged: {
    label: 'Flagged',
    hint: 'Stored, but flagged for review',
    cls: 'border-white/30 bg-white/[0.06] text-white/85',
    dot: 'bg-white/70',
    Icon: AlertCircle,
  },
  duplicate: {
    label: 'Duplicate',
    hint: 'Same content already processed — skipped',
    cls: 'border-white/10 bg-white/[0.03] text-white/40',
    dot: 'bg-white/25',
    Icon: Copy,
  },
  rejected: {
    label: 'Rejected',
    hint: 'Not a document (recoverable — in rejected/)',
    cls: 'border-white/30 bg-white/[0.06] text-white/85',
    dot: 'bg-white/70',
    Icon: X,
  },
  failed: {
    label: 'Failed',
    hint: 'Something went wrong — in failed/',
    cls: 'border-white/30 bg-white/[0.06] text-white/85',
    dot: 'bg-white/70',
    Icon: AlertCircle,
  },
};

// ── Page ──────────────────────────────────────────────────────────────────────
export default function DocumentsPage() {
  const [folders, setFolders] = useState<FolderData>({
    incoming: [],
    processing: [],
    processed: [],
    failed: [],
    rejected: [],
  });
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [online, setOnline] = useState(true);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  // ── Desktop notifications ──
  type Perm = NotificationPermission | 'unsupported';
  const [notifPerm, setNotifPerm] = useState<Perm>('default');
  const [notifOn, setNotifOn] = useState(false);
  const seenTerminal = useRef<Set<string> | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      setNotifPerm('unsupported');
      return;
    }
    setNotifPerm(Notification.permission);
    if (Notification.permission === 'granted') setNotifOn(true);
  }, []);

  const toggleNotif = useCallback(async () => {
    if (!('Notification' in window)) return;
    if (Notification.permission === 'granted') {
      setNotifOn((v) => !v);
      return;
    }
    if (Notification.permission === 'denied') {
      setToast('Notifications are blocked — enable them for this site in your browser settings.');
      setTimeout(() => setToast(null), 4500);
      return;
    }
    const perm = await Notification.requestPermission();
    setNotifPerm(perm);
    setNotifOn(perm === 'granted');
    if (perm === 'granted') {
      new Notification('Notifications on', {
        body: "You'll get a toast here whenever a document finishes processing.",
        icon: NOTIF_ICON,
      });
    }
  }, []);

  // ── Polling ──
  const poll = useCallback(async () => {
    try {
      const [fRes, dRes] = await Promise.all([
        fetch('/api/files', { cache: 'no-store' }),
        fetch('/api/documents', { cache: 'no-store' }),
      ]);
      if (!fRes.ok) throw new Error('files api');
      setFolders(await fRes.json());
      const dJson = await dRes.json();
      setDocuments(dJson.documents ?? []);
      setOnline(true);
      setHasLoaded(true);
    } catch {
      setOnline(false);
    }
  }, []);

  useEffect(() => {
    poll();
    const t = setInterval(poll, POLL_MS);
    return () => clearInterval(t);
  }, [poll]);

  // ── Upload ──
  const upload = useCallback(
    async (files: FileList | File[]) => {
      const list = Array.from(files);
      if (list.length === 0) return;
      setUploading(true);
      try {
        const fd = new FormData();
        list.forEach((f) => fd.append('files', f));
        const res = await fetch('/api/upload', { method: 'POST', body: fd });
        const json = await res.json();
        const n = (json.saved ?? []).length;
        setToast(n > 0 ? `${n} file${n > 1 ? 's' : ''} uploaded — watch it move through the pipeline` : 'Upload failed');
        poll();
      } catch {
        setToast('Upload failed — is the app running next to the doc folder?');
      } finally {
        setUploading(false);
        setTimeout(() => setToast(null), 4000);
      }
    },
    [poll],
  );

  // ── Journey: merge folder reality + Postgres records ──
  const journey = useMemo<JourneyItem[]>(() => {
    const byName = new Map<string, JourneyItem>();
    for (const d of documents) {
      const stage: Stage =
        d.decision === 'flagged' || d.status === 'flagged'
          ? 'flagged'
          : d.decision === 'duplicate' || d.status === 'duplicate'
            ? 'duplicate'
            : 'stored';
      if (d.status === 'processing') continue;
      byName.set(d.file_name, { name: d.file_name, stage, time: new Date(d.created_at).getTime(), doc: d });
    }
    for (const f of folders.incoming) byName.set(f.name, { name: f.name, stage: 'queued', time: f.mtime, size: f.size });
    for (const f of folders.processing)
      byName.set(f.name, {
        name: f.name,
        stage: Date.now() - f.mtime > STALL_AFTER_MS ? 'stalled' : 'analyzing',
        time: f.mtime,
        size: f.size,
      });
    for (const f of folders.rejected)
      byName.set(f.name, { name: f.name, stage: 'rejected', time: f.mtime, size: f.size });
    for (const f of folders.failed) {
      const existing = byName.get(f.name);
      byName.set(f.name, { ...(existing ?? { name: f.name }), name: f.name, stage: 'failed', time: f.mtime, size: f.size });
    }
    const order: Record<Stage, number> = { analyzing: 0, stalled: 1, queued: 2, failed: 3, rejected: 4, flagged: 5, stored: 6, duplicate: 7 };
    return [...byName.values()].sort((a, b) => order[a.stage] - order[b.stage] || b.time - a.time);
  }, [folders, documents]);

  const counts = useMemo(() => {
    const c = { active: 0, stored: 0, attention: 0 };
    for (const j of journey) {
      if (j.stage === 'queued' || j.stage === 'analyzing') c.active++;
      else if (j.stage === 'stored' || j.stage === 'duplicate' || j.stage === 'flagged') c.stored++;
      if (j.stage === 'failed' || j.stage === 'rejected' || j.stage === 'flagged' || j.stage === 'stalled') c.attention++;
    }
    return c;
  }, [journey]);

  // Desktop toast when a document first reaches a terminal stage.
  useEffect(() => {
    if (!hasLoaded) return;
    const TERMINAL: Stage[] = ['stored', 'flagged', 'rejected', 'failed', 'stalled'];
    const terminal = journey.filter((j) => TERMINAL.includes(j.stage));
    const keys = new Set(terminal.map((j) => `${j.name}::${j.stage}`));
    if (seenTerminal.current === null) {
      seenTerminal.current = keys;
      return;
    }
    if (notifOn && notifPerm === 'granted') {
      for (const j of terminal) {
        const key = `${j.name}::${j.stage}`;
        if (seenTerminal.current.has(key)) continue;
        const s = STAGE[j.stage];
        const type = j.doc?.document_type ? ` · ${j.doc.document_type}` : '';
        const conf = j.doc?.confidence != null ? ` · ${Math.round(Number(j.doc.confidence) * 100)}%` : '';
        try {
          new Notification(`${s.label}: ${j.name}`, { body: `${s.hint}${type}${conf}`, icon: NOTIF_ICON, tag: key });
        } catch {
          /* permission can be revoked mid-session */
        }
      }
    }
    seenTerminal.current = keys;
  }, [journey, notifOn, notifPerm, hasLoaded]);

  const notifActive = notifOn && notifPerm === 'granted';

  return (
    <div className="mx-auto flex h-full w-full max-w-4xl flex-col px-6 py-6">
      {/* Page header */}
      <div className="flex shrink-0 items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-white">Document Processing</h1>
          <p className="text-xs text-white/40">CLIP triage on the NPU · VLM OCR on the GPU · agent extraction — live</p>
        </div>
        <div className="flex items-center gap-2">
          {notifPerm !== 'unsupported' && (
            <button
              onClick={toggleNotif}
              title={notifActive ? 'Notifications on' : 'Enable notifications'}
              className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[11px] transition ${
                notifActive
                  ? 'border-brand-500/40 bg-brand-500/10 text-brand-300'
                  : 'border-white/15 bg-white/[0.03] text-white/50 hover:text-white/80'
              }`}
            >
              {notifActive ? <Bell size={12} /> : <BellOff size={12} />}
              <span className="hidden sm:inline">{notifActive ? 'Alerts on' : 'Alerts off'}</span>
            </button>
          )}
          <span
            className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[11px] ${
              online ? 'border-brand-500/40 bg-brand-500/10 text-brand-300' : 'border-white/25 bg-white/[0.05] text-white/70'
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${online ? 'animate-pulse bg-brand-400' : 'bg-white/50'}`} />
            {online ? 'Pipeline live' : 'Backend offline'}
          </span>
        </div>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          upload(e.dataTransfer.files);
        }}
        onClick={() => fileInput.current?.click()}
        className={`group mt-4 flex shrink-0 cursor-pointer items-center gap-4 rounded-2xl border-2 border-dashed px-5 py-4 transition-all ${
          dragOver
            ? 'border-brand-400 bg-brand-500/10 shadow-[0_0_32px_rgba(108,36,240,0.25)]'
            : 'border-white/15 bg-white/[0.02] hover:border-brand-500/60 hover:bg-brand-500/5'
        }`}
      >
        <input
          ref={fileInput}
          type="file"
          multiple
          accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp"
          className="hidden"
          onChange={(e) => e.target.files && upload(e.target.files)}
        />
        <div
          className={`grid h-11 w-11 shrink-0 place-items-center rounded-xl transition ${
            dragOver
              ? 'bg-brand-500/25 text-brand-300'
              : 'bg-white/[0.05] text-white/50 group-hover:bg-brand-500/15 group-hover:text-brand-300'
          }`}
        >
          {uploading ? <Loader2 size={20} className="animate-spin" /> : <Upload size={20} />}
        </div>
        <div className="min-w-0">
          <p className="text-sm font-semibold text-white">{uploading ? 'Uploading…' : 'Drop a document — or click to browse'}</p>
          <p className="mt-0.5 text-xs text-white/40">PDF · PNG · JPG — processed locally. Nothing leaves this machine.</p>
        </div>
      </div>

      {/* Stat pills */}
      <div className="mt-3 flex shrink-0 gap-2 text-xs">
        <StatPill dot="bg-brand-400" label="In pipeline" value={counts.active} pulse={counts.active > 0} />
        <StatPill dot="bg-white" label="Stored" value={counts.stored} />
        <StatPill dot="bg-white/35" label="Needs attention" value={counts.attention} />
      </div>

      {/* Journey list */}
      <div className="glass mt-3 flex min-h-0 flex-1 flex-col p-4">
        <div className="flex shrink-0 items-center justify-between">
          <h2 className="text-sm font-semibold text-white/90">Documents</h2>
          <span className="text-[11px] text-white/35">live · folders + Postgres</span>
        </div>
        <ul className="mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
          {journey.length === 0 && (
            <li className="rounded-xl border border-dashed border-white/10 py-10 text-center text-xs text-white/30">
              Nothing yet — drop a document above to start.
            </li>
          )}
          {journey.map((j) => (
            <JourneyRow key={`${j.stage}-${j.name}`} item={j} />
          ))}
        </ul>
      </div>

      {toast && (
        <div className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2 animate-fade-up rounded-xl border border-brand-500/40 bg-black/95 px-4 py-2.5 text-sm text-white shadow-2xl shadow-brand-500/25 backdrop-blur">
          {toast}
        </div>
      )}
    </div>
  );
}

function StatPill({ dot, label, value, pulse }: { dot: string; label: string; value: number; pulse?: boolean }) {
  return (
    <div className="flex items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1.5">
      <span className={`h-2 w-2 rounded-full ${dot} ${pulse ? 'animate-pulse' : ''}`} />
      <span className="text-white/50">{label}</span>
      <span className="font-semibold text-white/90">{value}</span>
    </div>
  );
}

function JourneyRow({ item }: { item: JourneyItem }) {
  const [open, setOpen] = useState(false);
  const s = STAGE[item.stage];
  const d = item.doc;
  const conf = d?.confidence ?? d?.ocr_confidence;
  const expandable = Boolean(d);

  return (
    <li className="animate-fade-in">
      <div
        onClick={() => expandable && setOpen((o) => !o)}
        className={`rounded-xl border border-white/[0.06] bg-white/[0.02] px-3 py-2.5 transition ${
          expandable ? 'cursor-pointer hover:border-brand-500/30 hover:bg-white/[0.04]' : ''
        }`}
      >
        <div className="flex items-center gap-3">
          <span className={`relative mt-0.5 h-2 w-2 shrink-0 rounded-full ${s.dot} ${s.active ? 'animate-pulse-ring' : ''}`} />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <p className="truncate text-sm font-medium text-white/90" title={item.name}>
                {item.name}
              </p>
              {d?.document_type && (
                <span className="shrink-0 rounded bg-white/[0.06] px-1.5 py-0.5 text-[10px] capitalize text-white/50">
                  {d.document_type}
                </span>
              )}
            </div>
            <p className="mt-0.5 truncate text-[11px] text-white/35">{s.hint}</p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {conf != null && <span className="text-[11px] text-white/40">{Math.round(Number(conf) * 100)}%</span>}
            <span className={`flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${s.cls}`}>
              <s.Icon size={11} className={s.active ? 'animate-spin' : ''} />
              {s.label}
            </span>
            <span className="w-14 text-right text-[10px] text-white/25">{relTime(item.time)}</span>
            {expandable && (
              <ChevronDown size={14} className={`text-white/40 transition-transform ${open ? 'rotate-180' : ''}`} />
            )}
          </div>
        </div>

        {open && d && (
          <div className="mt-3 animate-fade-in space-y-2.5 border-t border-white/[0.08] pt-3 text-xs">
            {d.summary && <p className="leading-relaxed text-white/75">{d.summary}</p>}
            {d.reason && (
              <p className="text-white/40">
                <span className="text-white/60">Why:</span> {d.reason}
              </p>
            )}
            {conf != null && (
              <div className="flex items-center gap-2">
                <span className="w-24 shrink-0 text-white/40">Confidence</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.08]">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-brand-600 to-brand-400"
                    style={{ width: `${Math.round(Number(conf) * 100)}%` }}
                  />
                </div>
                <span className="w-9 text-right text-white/60">{Math.round(Number(conf) * 100)}%</span>
              </div>
            )}
            {d.fields && Object.keys(d.fields).length > 0 && (
              <div className="flex flex-wrap gap-1.5 pt-0.5">
                {Object.entries(d.fields)
                  .filter(([, v]) => v != null && v !== '' && typeof v !== 'object')
                  .slice(0, 8)
                  .map(([k, v]) => (
                    <span key={k} className="rounded-md border border-brand-500/20 bg-brand-500/[0.07] px-2 py-1 text-[11px] text-white/80">
                      <span className="text-brand-300/80">{k}: </span>
                      {String(v).slice(0, 28)}
                    </span>
                  ))}
              </div>
            )}
          </div>
        )}
      </div>
    </li>
  );
}
