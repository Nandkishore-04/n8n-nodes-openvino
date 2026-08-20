import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { FOLDERS, folderPath, ensureFolders, type Folder } from '@/lib/paths';

export const dynamic = 'force-dynamic';

export interface FileEntry {
  name: string;
  size: number;
  mtime: number;
}

async function listFolder(f: Folder): Promise<FileEntry[]> {
  const dir = folderPath(f);
  let names: string[] = [];
  try {
    names = await fs.readdir(dir);
  } catch {
    return [];
  }
  const out: FileEntry[] = [];
  for (const name of names) {
    if (name.startsWith('.')) continue; // skip dotfiles / .part temp uploads
    try {
      const st = await fs.stat(path.join(dir, name));
      if (!st.isFile()) continue;
      out.push({ name, size: st.size, mtime: st.mtimeMs });
    } catch {
      // file vanished between readdir and stat — skip
    }
  }
  out.sort((a, b) => b.mtime - a.mtime); // newest first
  return out;
}

export async function GET() {
  await ensureFolders();
  const entries = await Promise.all(FOLDERS.map((f) => listFolder(f)));
  const result: Record<string, FileEntry[]> = {};
  FOLDERS.forEach((f, i) => {
    result[f] = entries[i];
  });
  return NextResponse.json(result);
}
