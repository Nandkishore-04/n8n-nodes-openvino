import * as path from 'path';
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';

const PROTO_PATH = path.join(__dirname, 'kserve.proto');

export interface InferInputTensor {
	name: string;
	datatype: string;
	shape: number[];
	contents?: {
		int64_contents?: number[];
		fp32_contents?: number[];
		bytes_contents?: Buffer[];
	};
}

export interface InferOutputTensor {
	name: string;
	datatype: string;
	shape: number[];
	contents?: {
		fp32_contents?: number[];
		int64_contents?: number[];
	};
}

export interface ModelInferRequest {
	model_name: string;
	model_version?: string;
	id?: string;
	inputs: InferInputTensor[];
	outputs?: Array<{ name: string }>;
	raw_input_contents?: Buffer[];
}

export interface ModelInferResponse {
	model_name: string;
	model_version: string;
	id: string;
	outputs: InferOutputTensor[];
	raw_output_contents: Buffer[];
}

export interface GrpcMetadata {
	targetDevice?: string;
	performanceHint?: string;
}

export class OvmsGrpcClient {
	private service: grpc.Client | null = null;

	connect(host: string, port: number): void {
		const packageDef = protoLoader.loadSync(PROTO_PATH, {
			keepCase: true,
			longs: String,
			enums: String,
			defaults: true,
			oneofs: true,
		});
		const proto = grpc.loadPackageDefinition(packageDef) as any;
		this.service = new proto.inference.GRPCInferenceService(
			`${host}:${port}`,
			grpc.credentials.createInsecure(),
			{ 'grpc.keepalive_timeout_ms': 5000 },
		);
	}

	private get svc(): any {
		if (!this.service) throw new Error('OvmsGrpcClient: call connect() before making requests');
		return this.service;
	}

	private buildMetadata(meta?: GrpcMetadata): grpc.Metadata {
		const md = new grpc.Metadata();
		if (meta?.targetDevice) md.add('x-target-device', meta.targetDevice);
		if (meta?.performanceHint) md.add('x-performance-hint', meta.performanceHint);
		return md;
	}

	modelInfer(request: ModelInferRequest, meta?: GrpcMetadata): Promise<ModelInferResponse> {
		return new Promise((resolve, reject) => {
			this.svc.ModelInfer(request, this.buildMetadata(meta), (err: grpc.ServiceError | null, res: ModelInferResponse) => {
				if (err) return reject(this.wrapGrpcError(err));
				resolve(res);
			});
		});
	}

	modelReady(name: string, version = '', meta?: GrpcMetadata): Promise<boolean> {
		return new Promise((resolve, reject) => {
			this.svc.ModelReady({ name, version }, this.buildMetadata(meta), (err: grpc.ServiceError | null, res: { ready: boolean }) => {
				if (err) return reject(this.wrapGrpcError(err));
				resolve(res.ready);
			});
		});
	}

	serverMetadata(meta?: GrpcMetadata): Promise<{ name: string; version: string; extensions: string[] }> {
		return new Promise((resolve, reject) => {
			this.svc.ServerMetadata({}, this.buildMetadata(meta), (err: grpc.ServiceError | null, res: any) => {
				if (err) return reject(this.wrapGrpcError(err));
				resolve(res);
			});
		});
	}

	// Decode raw_output_contents bytes → float32 array (little-endian).
	decodeFloat32(raw: Buffer): number[] {
		const out: number[] = [];
		const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
		for (let i = 0; i < raw.byteLength; i += 4) {
			out.push(view.getFloat32(i, true));
		}
		return out;
	}

	// Decode raw_output_contents bytes → int64 array (little-endian, returned as number).
	decodeInt64(raw: Buffer): number[] {
		const out: number[] = [];
		const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
		for (let i = 0; i < raw.byteLength; i += 8) {
			out.push(Number(view.getBigInt64(i, true)));
		}
		return out;
	}

	private wrapGrpcError(err: grpc.ServiceError): Error {
		const code = err.code;
		const msg = err.message ?? '';

		if (code === grpc.status.UNAVAILABLE || msg.includes('ECONNREFUSED')) {
			return new Error(`GRPC_UNAVAILABLE: ${err.details || msg}`);
		}
		if (code === grpc.status.NOT_FOUND) {
			return new Error(`GRPC_NOT_FOUND: ${err.details || msg}`);
		}
		if (code === grpc.status.INVALID_ARGUMENT) {
			return new Error(`GRPC_INVALID_ARGUMENT: ${err.details || msg}`);
		}
		if (code === grpc.status.DEADLINE_EXCEEDED) {
			return new Error(`GRPC_TIMEOUT: ${err.details || msg}`);
		}
		if (code === grpc.status.RESOURCE_EXHAUSTED) {
			return new Error(`GRPC_OVERLOADED: ${err.details || msg}`);
		}
		return new Error(`GRPC_ERROR(${code}): ${err.details || msg}`);
	}

	close(): void {
		if (this.service) grpc.closeClient(this.service);
		this.service = null;
	}
}
