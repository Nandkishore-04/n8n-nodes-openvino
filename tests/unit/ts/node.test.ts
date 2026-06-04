import { OpenVinoModelServer } from '../../../nodes/OpenVinoModelServer/OpenVinoModelServer.node';

// Mock the gRPC client so no real connection is made.
const mockModelInfer = jest.fn();
const mockModelReady = jest.fn();
const mockServerMetadata = jest.fn();
const mockClose = jest.fn();
const mockConnect = jest.fn();

jest.mock('../../../nodes/OpenVinoModelServer/grpc/client', () => ({
	OvmsGrpcClient: jest.fn().mockImplementation(() => ({
		connect: mockConnect,
		modelInfer: mockModelInfer,
		modelReady: mockModelReady,
		serverMetadata: mockServerMetadata,
		close: mockClose,
	})),
}));

const CREDS = {
	gatewayUrl: 'http://gateway:8000',
	llmServerUrl: 'http://ovms-llm:8000',
	grpcHost: 'ovms',
	grpcPort: 9000,
	apiKey: '',
};

// Build a fake IExecuteFunctions with the given params + a mock httpRequest.
function makeCtx(params: Record<string, unknown>, httpResponse: unknown, opts: { continueOnFail?: boolean } = {}) {
	const httpRequest = jest.fn().mockResolvedValue(httpResponse);
	const ctx: any = {
		getInputData: () => [{ json: {} }],
		getNodeParameter: (name: string) => params[name],
		getCredentials: jest.fn().mockResolvedValue(CREDS),
		continueOnFail: () => opts.continueOnFail ?? false,
		getNode: () => ({ name: 'OpenVINO Model Server' }),
		logger: { info: jest.fn(), debug: jest.fn(), warn: jest.fn(), error: jest.fn() },
		helpers: { httpRequest },
	};
	return { ctx, httpRequest };
}

const node = new OpenVinoModelServer();

beforeEach(() => {
	jest.clearAllMocks();
});

describe('OpenVinoModelServer.execute — predict (REST)', () => {
	it('builds the correct gateway URL and attaches device headers', async () => {
		const { ctx, httpRequest } = makeCtx(
			{
				operation: 'predict',
				transport: 'rest',
				modelName: 'text-classifier',
				inputData: '{"text":"hello"}',
				targetDevice: 'AUTO:NPU,GPU,CPU',
				performanceHint: 'THROUGHPUT',
				timeout: 30,
			},
			{ label: 'POSITIVE', confidence: 0.99 },
		);

		const result = await node.execute.call(ctx);

		expect(httpRequest).toHaveBeenCalledTimes(1);
		const callArg = httpRequest.mock.calls[0][0];
		expect(callArg.method).toBe('POST');
		expect(callArg.url).toBe('http://gateway:8000/v1/models/text-classifier:predict');
		expect(callArg.headers['X-Target-Device']).toBe('AUTO:NPU,GPU,CPU');
		expect(callArg.headers['X-Performance-Hint']).toBe('THROUGHPUT');
		expect(callArg.timeout).toBe(30000);
		expect(callArg.body).toEqual({ text: 'hello' });
		expect(result[0][0].json).toEqual({ label: 'POSITIVE', confidence: 0.99 });
	});

	it('parses inputData when it arrives as an object instead of a string', async () => {
		const { ctx, httpRequest } = makeCtx(
			{
				operation: 'predict',
				transport: 'rest',
				modelName: 'text-classifier',
				inputData: { text: 'hi' },
				targetDevice: 'AUTO',
				performanceHint: 'LATENCY',
				timeout: 60,
			},
			{ ok: true },
		);

		await node.execute.call(ctx);
		expect(httpRequest.mock.calls[0][0].body).toEqual({ text: 'hi' });
	});
});

describe('OpenVinoModelServer.execute — predict (gRPC)', () => {
	it('calls modelInfer with the parsed inputs and device metadata', async () => {
		mockModelInfer.mockResolvedValue({ outputs: [], raw_output_contents: [] });
		const { ctx } = makeCtx(
			{
				operation: 'predict',
				transport: 'grpc',
				modelName: 'text-classifier',
				inputData: '{"inputs":[{"name":"input_ids","datatype":"INT64","shape":[1,1],"contents":{"int64_contents":[101]}}]}',
				targetDevice: 'NPU',
				performanceHint: 'LATENCY',
				timeout: 60,
			},
			null,
		);

		await node.execute.call(ctx);

		expect(mockConnect).toHaveBeenCalledWith('ovms', 9000);
		expect(mockModelInfer).toHaveBeenCalledTimes(1);
		const [req, meta] = mockModelInfer.mock.calls[0];
		expect(req.model_name).toBe('text-classifier');
		expect(req.inputs).toHaveLength(1);
		expect(meta).toEqual({ targetDevice: 'NPU', performanceHint: 'LATENCY' });
		expect(mockClose).toHaveBeenCalled();
	});
});

describe('OpenVinoModelServer.execute — listModels & modelStatus (REST)', () => {
	it('listModels hits /v1/models', async () => {
		const { ctx, httpRequest } = makeCtx(
			{ operation: 'listModels', transport: 'rest', targetDevice: 'AUTO', performanceHint: 'LATENCY', timeout: 60 },
			{ models: ['text-classifier'] },
		);
		await node.execute.call(ctx);
		expect(httpRequest.mock.calls[0][0].url).toBe('http://gateway:8000/v1/models');
	});

	it('modelStatus hits /v1/models/{name}', async () => {
		const { ctx, httpRequest } = makeCtx(
			{ operation: 'modelStatus', transport: 'rest', modelName: 'text-classifier', targetDevice: 'AUTO', performanceHint: 'LATENCY', timeout: 60 },
			{ ready: true },
		);
		await node.execute.call(ctx);
		expect(httpRequest.mock.calls[0][0].url).toBe('http://gateway:8000/v1/models/text-classifier');
	});
});

describe('OpenVinoModelServer.execute — error handling', () => {
	it('throws NodeOperationError with a friendly message when continueOnFail is off', async () => {
		const { ctx } = makeCtx(
			{ operation: 'predict', transport: 'rest', modelName: 'text-classifier', inputData: '{"text":"x"}', targetDevice: 'AUTO', performanceHint: 'LATENCY', timeout: 60 },
			null,
		);
		ctx.helpers.httpRequest = jest.fn().mockRejectedValue(new Error('connect ECONNREFUSED 127.0.0.1:8000'));

		await expect(node.execute.call(ctx)).rejects.toThrow(/gateway/);
	});

	it('returns the friendly error in json when continueOnFail is on', async () => {
		const { ctx } = makeCtx(
			{ operation: 'predict', transport: 'rest', modelName: 'text-classifier', inputData: '{"text":"x"}', targetDevice: 'AUTO', performanceHint: 'LATENCY', timeout: 60 },
			null,
			{ continueOnFail: true },
		);
		ctx.helpers.httpRequest = jest.fn().mockRejectedValue(new Error('GRPC_NOT_FOUND: missing'));

		const result = await node.execute.call(ctx);
		expect(result[0][0].json.error).toContain("Model 'text-classifier' not found");
	});
});

describe('OpenVinoModelServer.execute — stubs', () => {
	it('returns a stub marker for documentInference', async () => {
		const { ctx } = makeCtx(
			{ operation: 'documentInference', targetDevice: 'AUTO', performanceHint: 'LATENCY', timeout: 60 },
			null,
		);
		const result = await node.execute.call(ctx);
		expect(result[0][0].json._stub).toBe(true);
	});
});
