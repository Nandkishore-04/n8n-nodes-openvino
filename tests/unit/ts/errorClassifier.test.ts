import { classifyOvmsError } from '../../../nodes/OpenVinoModelServer/OpenVinoModelServer.node';

const baseCtx = {
	transport: 'rest',
	modelName: 'text-classifier',
	gatewayUrl: 'http://gateway:8000',
	grpcHost: 'ovms',
	grpcPort: 9000,
};

describe('classifyOvmsError', () => {
	it('maps gRPC connection failure to a container hint (gRPC transport)', () => {
		const msg = classifyOvmsError(new Error('GRPC_UNAVAILABLE: connection refused'), {
			...baseCtx,
			transport: 'grpc',
		});
		expect(msg).toContain('ovms:9000');
		expect(msg).toContain('podman-compose up -d ovms');
	});

	it('maps a raw ECONNREFUSED to the gateway hint (REST transport)', () => {
		const msg = classifyOvmsError(new Error('connect ECONNREFUSED 127.0.0.1:8000'), baseCtx);
		expect(msg).toContain('gateway');
		expect(msg).toContain('podman-compose up -d gateway');
	});

	it('maps model-not-found to a config.json hint', () => {
		const msg = classifyOvmsError(new Error('GRPC_NOT_FOUND: model missing'), baseCtx);
		expect(msg).toContain("Model 'text-classifier' not found");
		expect(msg).toContain('config.json');
	});

	it('maps invalid argument to a tensor-shape hint', () => {
		const msg = classifyOvmsError(
			new Error('GRPC_INVALID_ARGUMENT: Invalid number of inputs - Expected: 2; Actual: 0'),
			baseCtx,
		);
		expect(msg).toContain('Invalid input');
		expect(msg).toContain('shapes');
	});

	it('maps timeout to a loading/overloaded hint', () => {
		const msg = classifyOvmsError(new Error('ETIMEDOUT'), baseCtx);
		expect(msg).toContain('timed out');
		expect(msg).toContain('podman logs ovms');
	});

	it('maps resource exhausted to an overloaded hint', () => {
		const msg = classifyOvmsError(new Error('GRPC_OVERLOADED: too many requests'), baseCtx);
		expect(msg).toContain('overloaded');
	});

	it('maps 503 / not ready to a wait hint', () => {
		const msg = classifyOvmsError(new Error('503 model_not_ready'), baseCtx);
		expect(msg).toContain('not ready');
	});

	it('maps device-unavailable to a /dev/dri hint', () => {
		const msg = classifyOvmsError(new Error('device NPU unavailable'), baseCtx);
		expect(msg).toContain('/dev/dri');
		expect(msg).toContain('RENDER_GROUP_ID');
	});

	it('maps 401 to an API key hint', () => {
		const msg = classifyOvmsError(new Error('401 Unauthorized'), baseCtx);
		expect(msg).toContain('API Key');
	});

	it('falls back to the raw message for unknown errors', () => {
		const msg = classifyOvmsError(new Error('some weird unexpected failure'), baseCtx);
		expect(msg).toBe('some weird unexpected failure');
	});
});
