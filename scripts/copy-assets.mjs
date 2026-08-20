// Copies node assets (icons, protobufs) into dist/ after tsc.
//
// This used to be `find ... -exec cp --parents`, which is Unix-only: on Windows `find` is a
// different command entirely, so the build printed an error and the node icon never made it
// into dist/. Node's own fs works the same everywhere.
import { cp, mkdir, readdir } from 'node:fs/promises';
import { dirname, join, relative } from 'node:path';

const SRC = 'nodes';
const OUT = 'dist';
const EXT = ['.svg', '.proto'];

async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walk(full)));
    else if (EXT.some((e) => entry.name.endsWith(e))) out.push(full);
  }
  return out;
}

const files = await walk(SRC);
for (const file of files) {
  const dest = join(OUT, relative('.', file));
  await mkdir(dirname(dest), { recursive: true });
  await cp(file, dest);
}
console.log(`copied ${files.length} asset${files.length === 1 ? '' : 's'} to ${OUT}/`);
