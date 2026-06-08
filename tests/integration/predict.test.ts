// Integration tests — hit the LIVE stack (gateway + OVMS).
// Prereq: cd deployment && podman-compose up -d  (wait for ovms healthy)
// Run:    npm run test:integration
import { OvmsGrpcClient } from '../../nodes/OpenVinoModelServer/grpc/client';

const GATEWAY = process.env.OVMS_GATEWAY_URL ?? 'http://localhost:8000';
const GRPC_HOST = process.env.OVMS_GRPC_HOST ?? 'localhost';
const GRPC_PORT = Number(process.env.OVMS_GRPC_PORT ?? 9000);

describe('Integration — REST via gateway', () => {
	it('predicts POSITIVE sentiment for upbeat text', async () => {
		const res = await fetch(`${GATEWAY}/v1/models/text-classifier:predict`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ text: 'this product is amazing' }),
		});
		expect(res.status).toBe(200);

		const data = await res.json() as any;
		const pred = data.predictions[0];
		expect(pred.label).toBe('POSITIVE');
		expect(pred.confidence).toBeGreaterThan(0.9);
	});

	it('predicts NEGATIVE sentiment for downbeat text', async () => {
		const res = await fetch(`${GATEWAY}/v1/models/text-classifier:predict`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ text: 'this is terrible and broken' }),
		});
		const data = await res.json() as any;
		expect(data.predictions[0].label).toBe('NEGATIVE');
	});

	it('reports the gateway is healthy', async () => {
		const res = await fetch(`${GATEWAY}/health`);
		const data = await res.json() as any;
		expect(data.status).toBe('healthy');
		expect(data.models).toContain('text-classifier');
	});
});

describe('Integration — gRPC direct to OVMS', () => {
	let client: OvmsGrpcClient;

	beforeAll(() => {
		client = new OvmsGrpcClient();
		client.connect(GRPC_HOST, GRPC_PORT);
	});

	afterAll(() => client.close());

	it('returns server metadata', async () => {
		const meta = await client.serverMetadata();
		expect(meta.name).toContain('OpenVINO');
		expect(meta.version).toBeTruthy();
	});

	it('reports text-classifier is ready', async () => {
		await expect(client.modelReady('text-classifier')).resolves.toBe(true);
	});

	it('runs inference and returns FP32 logits', async () => {
		// [CLS] hello [SEP] — minimal valid DistilBERT input
		const res = await client.modelInfer({
			model_name: 'text-classifier',
			inputs: [
				{ name: 'input_ids',      datatype: 'INT64', shape: [1, 3], contents: { int64_contents: [101, 7592, 102] } },
				{ name: 'attention_mask', datatype: 'INT64', shape: [1, 3], contents: { int64_contents: [1, 1, 1] } },
			],
		});

		expect(res.raw_output_contents).toHaveLength(1);
		const logits = client.decodeFloat32(res.raw_output_contents[0]);
		expect(logits).toHaveLength(2);          // NEGATIVE, POSITIVE
		expect(typeof logits[0]).toBe('number');
	});
});
