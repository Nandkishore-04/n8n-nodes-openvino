import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { pool } from '@/lib/db';
import { WATCH_DIR, FOLDERS, folderPath } from '@/lib/paths';

export const dynamic = 'force-dynamic';

// Every link between this app and the backend, checked one by one. Each result carries
// the exact fix, because the failure modes here are all configuration, not code:
// the defaults in lib/paths.ts and lib/db.ts are the Linux/container values.
export interface Probe {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
  fix?: string;
}

const N8N_BASE = (process.env.N8N_BASE_URL || 'http://localhost:5678').replace(/\/+$/, '');
const RAG_URL = process.env.N8N_RAG_WEBHOOK_URL || `${N8N_BASE}/webhook/rag-query`;
const GATEWAY = (process.env.GATEWAY_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

/** A GET against a POST-only n8n webhook distinguishes three states cheaply — without
 *  actually running the workflow: registered (405-ish "did you mean POST"), not
 *  registered (workflow inactive), or n8n itself unreachable. */
async function probeWebhook(id: string, label: string, url: string): Promise<Probe> {
  try {
    const r = await fetch(url, { method: 'GET', cache: 'no-store', signal: AbortSignal.timeout(4000) });
    const body = (await r.text()).slice(0, 300);
    const registered = /did you mean to make a (post|get) request/i.test(body) || r.ok;
    if (registered) {
      return { id, label, ok: true, detail: `Registered and listening at ${url}` };
    }
    if (/not registered/i.test(body)) {
      return {
        id, label, ok: false,
        detail: `n8n is running but this webhook is not registered (${url})`,
        fix: 'Open that workflow in n8n and switch it to Active (top-right toggle).',
      };
    }
    return { id, label, ok: false, detail: `Unexpected response ${r.status}: ${body}`, fix: 'Check the workflow in n8n.' };
  } catch (e) {
    const msg = (e as Error).message;
    return {
      id, label, ok: false,
      detail: `Could not reach n8n at ${url} (${msg})`,
      fix: 'Start n8n (n8n start) and confirm it is on port 5678.',
    };
  }
}

async function probeWatchDir(): Promise<Probe> {
  const id = 'watch';
  const label = 'Document folder (upload target)';

  // Writable is NOT the same as correct. If WATCH_DIR is unset we fall back to the
  // dev default, which Node will happily create on any OS — uploads then land in a
  // folder WF1 never watches and sit there silently. Catch that explicitly.
  if (!process.env.WATCH_DIR) {
    return {
      id, label, ok: false,
      detail: `WATCH_DIR is not set — falling back to the built-in default (${WATCH_DIR}). Uploads would go somewhere the pipeline never looks.`,
      fix: 'Create web/.env.local with WATCH_DIR set to the same docRoot as the WF1 "Config" node, then restart npm run dev (env is read at startup).',
    };
  }

  try {
    await fs.mkdir(WATCH_DIR, { recursive: true });
    // Prove we can actually write — a read-only or wrong-drive path fails here, not at upload time.
    const probe = path.join(WATCH_DIR, '.write-probe');
    await fs.writeFile(probe, 'ok');
    await fs.rm(probe, { force: true });
    const counts = await Promise.all(
      FOLDERS.map(async (f) => {
        try {
          const names = await fs.readdir(folderPath(f));
          return `${f}:${names.filter((n) => !n.startsWith('.')).length}`;
        } catch {
          return `${f}:-`;
        }
      }),
    );
    return { id, label, ok: true, detail: `${WATCH_DIR} — ${counts.join('  ')}` };
  } catch (e) {
    return {
      id, label, ok: false,
      detail: `Cannot write to ${WATCH_DIR} (${(e as Error).message})`,
      fix: 'Set WATCH_DIR in web/.env.local to the same docRoot the WF1 Config node uses, e.g. WATCH_DIR=C:/Users/devcloud/proj-demo',
    };
  }
}

async function probePostgres(): Promise<Probe> {
  const id = 'pg';
  const label = 'Postgres (processed documents)';
  try {
    const { rows } = await pool.query<{ n: string }>('SELECT count(*)::text AS n FROM pipeline_documents');
    return { id, label, ok: true, detail: `Connected — ${rows[0]?.n ?? '0'} rows in pipeline_documents` };
  } catch (e) {
    const msg = (e as Error).message;
    let fix =
      'Set PGUSER / PGPASSWORD / PGDATABASE in web/.env.local — the defaults (n8n/n8npassword) are the container values, not your native Postgres.';
    if (/relation .* does not exist/i.test(msg)) {
      fix = 'Apply deployment/sql/init.sql to the n8n database (pgAdmin → Query Tool).';
    } else if (/ECONNREFUSED|ETIMEDOUT|EHOSTUNREACH/i.test(msg)) {
      fix = `Postgres is not answering on ${process.env.PGHOST || 'localhost'}:${process.env.PGPORT || 5432}. Start the PostgreSQL service, or set PGHOST/PGPORT in web/.env.local.`;
    } else if (/password authentication failed|no pg_hba|role .* does not exist/i.test(msg)) {
      fix = 'Wrong Postgres user or password — set PGUSER and PGPASSWORD in web/.env.local to the account you use in pgAdmin.';
    } else if (/database .* does not exist/i.test(msg)) {
      fix = 'Set PGDATABASE in web/.env.local to the database holding pipeline_documents (usually n8n).';
    }
    return { id, label, ok: false, detail: msg, fix };
  }
}

async function probeGateway(): Promise<Probe> {
  const id = 'gateway';
  const label = 'OpenVINO gateway (models)';
  try {
    const r = await fetch(`${GATEWAY}/health`, { cache: 'no-store', signal: AbortSignal.timeout(4000) });
    const j = (await r.json()) as Record<string, unknown>;
    const devices = Array.isArray(j.available_devices) ? (j.available_devices as string[]).join(', ') : 'unknown';
    const extras = [j.embeddings ? 'embeddings' : null, j.asr ? 'ASR' : null, j.tts ? 'TTS' : null]
      .filter(Boolean)
      .join(' · ');
    return { id, label, ok: true, detail: `Devices: ${devices}${extras ? ` — ${extras}` : ''}` };
  } catch (e) {
    return {
      id, label, ok: false,
      detail: `No response from ${GATEWAY} (${(e as Error).message})`,
      fix: 'Start scripts/native_gateway.py, or set GATEWAY_URL in web/.env.local.',
    };
  }
}

export async function GET() {
  const probes = await Promise.all([
    probeWatchDir(),
    probePostgres(),
    probeGateway(),
    probeWebhook('wf2', 'RAG workflow (Ask AI)', RAG_URL),
    probeWebhook('agent', 'Agent chatbot', `${N8N_BASE}/webhook/agent-chat`),
  ]);
  return NextResponse.json({ ok: probes.every((p) => p.ok), probes });
}
