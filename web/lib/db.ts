import { Pool } from 'pg';

// Pooled connection to the same Postgres the n8n stack uses (exposed on host :5432).
// Reads standard PG* env vars; defaults match deployment/podman-compose.yml + .env.
// A single pool is cached on the global object to survive Next.js dev hot-reloads.
const globalForPg = globalThis as unknown as { _pgPool?: Pool };

export const pool =
  globalForPg._pgPool ??
  new Pool({
    host: process.env.PGHOST || 'localhost',
    port: Number(process.env.PGPORT || 5432),
    database: process.env.PGDATABASE || 'n8n',
    user: process.env.PGUSER || 'n8n',
    password: process.env.PGPASSWORD || 'n8npassword',
    max: 4,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 3000,
  });

if (process.env.NODE_ENV !== 'production') globalForPg._pgPool = pool;
