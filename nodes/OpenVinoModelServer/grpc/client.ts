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
    );
  }

  private get svc(): any {
    if (!this.service) throw new Error('OvmsGrpcClient: call connect() before making requests');
    return this.service;
  }

  modelInfer(request: ModelInferRequest): Promise<ModelInferResponse> {
    return new Promise((resolve, reject) => {
      this.svc.ModelInfer(request, (err: grpc.ServiceError | null, res: ModelInferResponse) => {
        if (err) return reject(new Error(`gRPC ModelInfer failed: ${err.message}`));
        resolve(res);
      });
    });
  }

  modelReady(name: string, version = ''): Promise<boolean> {
    return new Promise((resolve, reject) => {
      this.svc.ModelReady({ name, version }, (err: grpc.ServiceError | null, res: { ready: boolean }) => {
        if (err) return reject(new Error(`gRPC ModelReady failed: ${err.message}`));
        resolve(res.ready);
      });
    });
  }

  serverMetadata(): Promise<{ name: string; version: string; extensions: string[] }> {
    return new Promise((resolve, reject) => {
      this.svc.ServerMetadata({}, (err: grpc.ServiceError | null, res: any) => {
        if (err) return reject(new Error(`gRPC ServerMetadata failed: ${err.message}`));
        resolve(res);
      });
    });
  }

  // Decode raw_output_contents bytes → float32 array (little-endian).
  // OVMS returns raw bytes for FP32 outputs when raw_output_contents is populated.
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

  close(): void {
    if (this.service) grpc.closeClient(this.service);
    this.service = null;
  }
}
