import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { folderPath, ensureFolders } from '@/lib/paths';

export const dynamic = 'force-dynamic';

// n8n webhook that kicks off WF1 immediately on upload (instead of waiting for the
// 30s schedule poll). Needs the workflow Active. Falls back silently to the schedule
// safety net if unreachable (e.g. workflow not published).
const N8N_WEBHOOK_URL =
  process.env.N8N_WEBHOOK_URL || 'http://localhost:5678/webhook/process-document';

// Sanitize to a safe basename (no path traversal, no exotic chars).
function safeName(name: string): string {
  const base = path.basename(name);
  return base.replace(/[^\w.\- ]/g, '_').slice(0, 200);
}

// Fire-and-forget kick to the n8n webhook for one claimed filename.
async function triggerWorkflow(fileName: string): Promise<void> {
  try {
    await fetch(N8N_WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fileName }),
      signal: AbortSignal.timeout(3000),
    });
  } catch {
    // Workflow inactive/unreachable — the Schedule Trigger will pick the file up.
  }
}

export async function POST(req: NextRequest) {
  await ensureFolders();
  const incoming = folderPath('incoming');

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return NextResponse.json({ error: 'Expected multipart/form-data.' }, { status: 400 });
  }

  const files = form.getAll('files').filter((f): f is File => f instanceof File);
  if (files.length === 0) {
    return NextResponse.json({ error: 'No files in request.' }, { status: 400 });
  }

  const saved: string[] = [];
  const errors: { name: string; error: string }[] = [];

  for (const file of files) {
    const name = safeName(file.name || 'document');
    // Write to a hidden temp file, then atomically rename into place. The dot-prefix
    // means the n8n Claim File node ignores it until it's a complete, renamed file —
    // so the pipeline never grabs a half-uploaded document.
    const tmp = path.join(incoming, `.${name}.part`);
    const dst = path.join(incoming, name);
    try {
      const buf = Buffer.from(await file.arrayBuffer());
      await fs.writeFile(tmp, buf);
      await fs.rename(tmp, dst);
      saved.push(name);
    } catch (e) {
      await fs.rm(tmp, { force: true }).catch(() => {});
      errors.push({ name, error: (e as Error).message });
    }
  }

  // Kick the workflow once per saved file (each call claims that exact file).
  await Promise.allSettled(saved.map(triggerWorkflow));

  return NextResponse.json({ saved, errors });
}
