// Server-side proxy for the gateway's /health (the gateway sends no CORS headers,
// so the browser can't call it directly). Lets the home page show live chip status.
export const dynamic = 'force-dynamic';

const GATEWAY = process.env.GATEWAY_URL ?? 'http://127.0.0.1:8000';

export async function GET() {
  try {
    const r = await fetch(`${GATEWAY}/health`, { cache: 'no-store', signal: AbortSignal.timeout(2500) });
    const json = await r.json();
    return Response.json({ online: true, ...json });
  } catch {
    return Response.json({ online: false, available_devices: [] });
  }
}
