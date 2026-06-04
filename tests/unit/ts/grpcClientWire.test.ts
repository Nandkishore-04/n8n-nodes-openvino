// Covers the gRPC wire methods (modelInfer/modelReady/serverMetadata),
// metadata building, and error wrapping — by faking @grpc/grpc-js.

// Holds the behavior each fake service method should exhibit per-test.
const serviceBehavior: {
	err: any;
	res: any;
} = { err: null, res: {} };

// Captures metadata passed into the last call so tests can assert on it.
let lastMetadataAdds: Array<[string, string]> = [];

class FakeMetadata {
	add(key: string, value: string) {
		lastMetadataAdds.push([key, value]);
	}
}

function fakeMethod(_req: any, _meta: any, cb: (err: any, res: any) => void) {
	cb(serviceBehavior.err, serviceBehavior.res);
}

class FakeService {
	ModelInfer = fakeMethod;
	ModelReady = fakeMethod;
	ServerMetadata = fakeMethod;
}

jest.mock('@grpc/proto-loader', () => ({
	loadSync: jest.fn(() => ({})),
}));

jest.mock('@grpc/grpc-js', () => ({
	loadPackageDefinition: jest.fn(() => ({
		inference: { GRPCInferenceService: FakeService },
	})),
	credentials: { createInsecure: jest.fn(() => ({})) },
	Metadata: FakeMetadata,
	closeClient: jest.fn(),
	status: {
		OK: 0,
		NOT_FOUND: 5,
		INVALID_ARGUMENT: 3,
		DEADLINE_EXCEEDED: 4,
		RESOURCE_EXHAUSTED: 8,
		UNAVAILABLE: 14,
	},
}));

import * as grpc from '@grpc/grpc-js';
import { OvmsGrpcClient } from '../../../nodes/OpenVinoModelServer/grpc/client';

function connected(): OvmsGrpcClient {
	const c = new OvmsGrpcClient();
	c.connect('ovms', 9000);
	return c;
}

beforeEach(() => {
	serviceBehavior.err = null;
	serviceBehavior.res = {};
	lastMetadataAdds = [];
});

describe('OvmsGrpcClient wire methods (success)', () => {
	it('modelInfer resolves with the response', async () => {
		serviceBehavior.res = { model_name: 'm', outputs: [], raw_output_contents: [] };
		const c = connected();
		await expect(c.modelInfer({ model_name: 'm', inputs: [] })).resolves.toEqual(serviceBehavior.res);
	});

	it('modelReady resolves the ready boolean', async () => {
		serviceBehavior.res = { ready: true };
		const c = connected();
		await expect(c.modelReady('text-classifier')).resolves.toBe(true);
	});

	it('serverMetadata resolves the metadata object', async () => {
		serviceBehavior.res = { name: 'OVMS', version: '2026.1', extensions: [] };
		const c = connected();
		await expect(c.serverMetadata()).resolves.toEqual(serviceBehavior.res);
	});

	it('attaches target device + performance hint as metadata', async () => {
		serviceBehavior.res = { ready: true };
		const c = connected();
		await c.modelReady('m', '', { targetDevice: 'NPU', performanceHint: 'LATENCY' });
		expect(lastMetadataAdds).toEqual([
			['x-target-device', 'NPU'],
			['x-performance-hint', 'LATENCY'],
		]);
	});
});

describe('OvmsGrpcClient error wrapping', () => {
	const cases: Array<[number, string, RegExp]> = [
		[grpc.status.UNAVAILABLE, 'down', /GRPC_UNAVAILABLE/],
		[grpc.status.NOT_FOUND, 'no model', /GRPC_NOT_FOUND/],
		[grpc.status.INVALID_ARGUMENT, 'bad shape', /GRPC_INVALID_ARGUMENT/],
		[grpc.status.DEADLINE_EXCEEDED, 'slow', /GRPC_TIMEOUT/],
		[grpc.status.RESOURCE_EXHAUSTED, 'busy', /GRPC_OVERLOADED/],
		[99, 'weird', /GRPC_ERROR\(99\)/],
	];

	it.each(cases)('maps gRPC status %i to a typed error', async (code, details, expected) => {
		serviceBehavior.err = { code, details, message: details };
		const c = connected();
		await expect(c.modelInfer({ model_name: 'm', inputs: [] })).rejects.toThrow(expected);
	});
});

describe('OvmsGrpcClient.close', () => {
	it('calls grpc.closeClient and clears the service', () => {
		const c = connected();
		c.close();
		expect((grpc as any).closeClient).toHaveBeenCalled();
	});
});
