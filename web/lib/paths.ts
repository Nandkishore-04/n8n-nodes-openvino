import path from 'path';
import fs from 'fs/promises';

// The shared watch folder. Same physical dir the n8n container mounts at /data/proj-demo.
// Override with WATCH_DIR env var when deploying elsewhere (e.g. Lunar Lake / WSL2).
export const WATCH_DIR = process.env.WATCH_DIR || '/home/nandyy/Documents/proj-demo';

export const FOLDERS = ['incoming', 'processing', 'processed', 'failed', 'rejected'] as const;
export type Folder = (typeof FOLDERS)[number];

export const folderPath = (f: Folder) => path.join(WATCH_DIR, f);

/** Ensure all pipeline folders exist (idempotent). */
export async function ensureFolders(): Promise<void> {
  await Promise.all(FOLDERS.map((f) => fs.mkdir(folderPath(f), { recursive: true })));
}
