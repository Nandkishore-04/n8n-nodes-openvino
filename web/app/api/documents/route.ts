import { NextResponse } from 'next/server';
import { pool } from '@/lib/db';

export const dynamic = 'force-dynamic';

export interface DocumentRow {
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

export async function GET() {
  try {
    const { rows } = await pool.query<DocumentRow>(
      `SELECT id, file_name, document_type, decision, reason, confidence,
              ocr_confidence, fields, summary, status, created_at
         FROM pipeline_documents
        ORDER BY created_at DESC
        LIMIT 100`,
    );
    return NextResponse.json({ documents: rows });
  } catch (e) {
    // Table missing / DB unreachable → empty list, so the UI degrades gracefully.
    return NextResponse.json({ documents: [], error: (e as Error).message }, { status: 200 });
  }
}
