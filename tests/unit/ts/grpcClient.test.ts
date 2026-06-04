import { OvmsGrpcClient } from '../../../nodes/OpenVinoModelServer/grpc/client';

describe('OvmsGrpcClient — tensor decoding', () => {
	let client: OvmsGrpcClient;

	beforeEach(() => {
		client = new OvmsGrpcClient();
	});

	describe('decodeFloat32', () => {
		it('decodes the validated text-classifier logits', () => {
			// These are the exact bytes OVMS returned for "this product is amazing"
			// → logits [-3.6784689, 3.9597495]
			const raw = Buffer.from([9, 108, 107, 192, 137, 108, 125, 64]);
			const out = client.decodeFloat32(raw);

			expect(out).toHaveLength(2);
			expect(out[0]).toBeCloseTo(-3.6784689, 4);
			expect(out[1]).toBeCloseTo(3.9597495, 4);
		});

		it('decodes a single float', () => {
			// 1.0 in little-endian float32 = 00 00 80 3f
			const raw = Buffer.from([0x00, 0x00, 0x80, 0x3f]);
			expect(client.decodeFloat32(raw)).toEqual([1]);
		});

		it('returns empty array for empty buffer', () => {
			expect(client.decodeFloat32(Buffer.alloc(0))).toEqual([]);
		});
	});

	describe('decodeInt64', () => {
		it('decodes little-endian int64 values', () => {
			// 101 and 102 as int64 LE
			const raw = Buffer.alloc(16);
			raw.writeBigInt64LE(BigInt(101), 0);
			raw.writeBigInt64LE(BigInt(102), 8);
			expect(client.decodeInt64(raw)).toEqual([101, 102]);
		});

		it('returns empty array for empty buffer', () => {
			expect(client.decodeInt64(Buffer.alloc(0))).toEqual([]);
		});
	});

	describe('connect guard', () => {
		it('throws a clear error if a request is made before connect()', async () => {
			await expect(client.modelReady('text-classifier')).rejects.toThrow(
				/call connect\(\) before making requests/,
			);
		});
	});
});
