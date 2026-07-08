-- WF1 metadata + audit tables.
-- Live in the existing `n8n` database (n8n ignores tables it doesn't own); prefixed
-- `pipeline_` to avoid any clash with n8n's own schema.
--
-- Apply once (the postgres volume already has data, so docker-entrypoint-initdb.d
-- scripts will NOT auto-run on an existing volume):
--   podman exec -i n8n-postgres psql -U n8n -d n8n < deployment/sql/init.sql

CREATE TABLE IF NOT EXISTS pipeline_documents (
  id             SERIAL PRIMARY KEY,
  file_hash      TEXT UNIQUE NOT NULL,               -- sha256 of raw bytes = dedup key
  file_name      TEXT NOT NULL,
  mime_type      TEXT,                               -- captured at the gate, before OCR
  file_size      BIGINT,                             -- bytes, captured at the gate
  document_type  TEXT,
  decision       TEXT,                               -- enriched | flagged | duplicate
  reason         TEXT,
  confidence     NUMERIC,                            -- agent confidence
  ocr_confidence NUMERIC,                            -- OCR aggregate confidence
  fields         JSONB,                              -- structured extracted fields
  summary        TEXT,
  status         TEXT NOT NULL DEFAULT 'processing', -- processing|processed|flagged|duplicate|reviewed|failed
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pipeline_audit (
  id           SERIAL PRIMARY KEY,
  document_id  INTEGER REFERENCES pipeline_documents(id),
  file_name    TEXT,
  action       TEXT NOT NULL,                        -- processed|flagged|reviewed|duplicate-skipped|corrected
  actor        TEXT NOT NULL DEFAULT 'system',
  details      JSONB,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_documents_created_at ON pipeline_documents (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_audit_document_id ON pipeline_audit (document_id);
