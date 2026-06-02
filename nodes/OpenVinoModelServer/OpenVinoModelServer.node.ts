import {
	IDataObject,
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

import { OvmsGrpcClient } from './grpc/client';

// ── Error classifier ──────────────────────────────────────────────────────────
// Inspects raw errors and returns a user-friendly message for n8n UI display.

function classifyOvmsError(err: Error, ctx: {
	transport: string;
	modelName?: string;
	gatewayUrl?: string;
	grpcHost?: string;
	grpcPort?: number;
}): string {
	const msg = err.message;
	const lower = msg.toLowerCase();

	// gRPC typed errors (prefixed by wrapGrpcError in client.ts)
	if (msg.startsWith('GRPC_UNAVAILABLE') || lower.includes('econnrefused')) {
		if (ctx.transport === 'grpc') {
			return `Cannot connect to OVMS gRPC at ${ctx.grpcHost}:${ctx.grpcPort}. Is the ovms container running? Run: podman-compose up -d ovms`;
		}
		return `Cannot connect to gateway at ${ctx.gatewayUrl}. Is ovms-gateway running? Run: podman-compose up -d gateway`;
	}

	if (msg.startsWith('GRPC_NOT_FOUND') || lower.includes('404') || lower.includes('not found')) {
		return `Model '${ctx.modelName}' not found on OVMS. Check deployment/config.json and confirm the model IR files exist under deployment/models/.`;
	}

	if (msg.startsWith('GRPC_INVALID_ARGUMENT') || lower.includes('invalid number of inputs') || lower.includes('invalid_argument')) {
		return `Invalid input for '${ctx.modelName}'. ${msg} — check tensor names, datatypes, and shapes match the model's expected inputs.`;
	}

	if (msg.startsWith('GRPC_TIMEOUT') || lower.includes('timeout') || lower.includes('etimedout') || lower.includes('deadline')) {
		return `Request timed out. Model '${ctx.modelName}' may still be loading or the container is overloaded. Check: podman logs ovms`;
	}

	if (msg.startsWith('GRPC_OVERLOADED') || lower.includes('resource_exhausted')) {
		return `OVMS is overloaded. Too many concurrent requests for model '${ctx.modelName}'. Reduce request rate or increase cache_size.`;
	}

	if (lower.includes('503') || lower.includes('not ready') || lower.includes('model_not_ready')) {
		return `Model '${ctx.modelName}' is not ready yet. Wait for OVMS to finish loading. Check: curl http://localhost:9001/v1/models/${ctx.modelName}`;
	}

	if (lower.includes('device') && (lower.includes('unavailable') || lower.includes('not supported'))) {
		return `OpenVINO device unavailable. Check /dev/dri permissions and RENDER_GROUP_ID in deployment/.env. Run: ls -la /dev/dri/`;
	}

	if (lower.includes('401') || lower.includes('unauthorized')) {
		return `Authentication failed. Check the API Key in your OpenVINO Model Server credentials.`;
	}

	// Fallback — return raw message so nothing is swallowed silently
	return msg;
}

// ── Node ──────────────────────────────────────────────────────────────────────

export class OpenVinoModelServer implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'OpenVINO Model Server',
		name: 'openVinoModelServer',
		icon: 'file:openvino.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Run AI inference via OpenVINO Model Server with GPU/NPU acceleration',
		defaults: { name: 'OpenVINO Model Server' },
		inputs: ['main'],
		outputs: ['main'],
		credentials: [{ name: 'openVinoModelServerApi', required: true }],
		properties: [

			// ── Operation ─────────────────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				options: [
					{ name: 'Predict',            value: 'predict',           description: 'Run inference on a classic model' },
					{ name: 'Document Inference', value: 'documentInference', description: 'PDF/image → OCR text + bounding boxes' },
					{ name: 'Embeddings',         value: 'embeddings',        description: 'Text → BGE-small-en vector' },
					{ name: 'Chat Completion',    value: 'chatCompletion',    description: 'Single LLM call via Qwen3-4B' },
					{ name: 'List Models',        value: 'listModels',        description: 'List all models served by OVMS' },
					{ name: 'Get Model Status',   value: 'modelStatus',       description: 'Per-model readiness check' },
				],
				default: 'predict',
			},

			// ── Transport ─────────────────────────────────────────────────────────
			{
				displayName: 'Transport',
				name: 'transport',
				type: 'options',
				options: [
					{ name: 'REST',  value: 'rest',  description: 'HTTP through the gateway — handles tokenization and pre/post processing' },
					{ name: 'gRPC', value: 'grpc', description: 'Direct binary protocol to OVMS port 9000 — you supply raw tensors' },
				],
				default: 'rest',
				displayOptions: {
					show: { operation: ['predict', 'listModels', 'modelStatus'] },
				},
			},

			// ── Model name ────────────────────────────────────────────────────────
			{
				displayName: 'Model Name',
				name: 'modelName',
				type: 'string',
				default: 'text-classifier',
				displayOptions: {
					show: { operation: ['predict', 'modelStatus'] },
				},
			},

			// ── Predict: input data ───────────────────────────────────────────────
			{
				displayName: 'Input Data',
				name: 'inputData',
				type: 'json',
				default: '{"text": "enter your text here"}',
				description: 'REST: {"text": "…"}. gRPC: KServe v2 inputs array — {"inputs":[{"name":"input_ids","datatype":"INT64","shape":[1,5],"contents":{"int64_contents":[…]}}]}',
				displayOptions: {
					show: { operation: ['predict'] },
				},
			},

			// ── AUTO plugin ───────────────────────────────────────────────────────
			{
				displayName: 'Target Device',
				name: 'targetDevice',
				type: 'options',
				options: [
					{ name: 'AUTO (let OpenVINO decide)',         value: 'AUTO' },
					{ name: 'AUTO: NPU → GPU → CPU',             value: 'AUTO:NPU,GPU,CPU' },
					{ name: 'AUTO: NPU → CPU',                   value: 'AUTO:NPU,CPU' },
					{ name: 'AUTO: GPU → CPU',                   value: 'AUTO:GPU,CPU' },
					{ name: 'NPU',                               value: 'NPU' },
					{ name: 'GPU',                               value: 'GPU' },
					{ name: 'CPU',                               value: 'CPU' },
				],
				default: 'AUTO',
				description: 'OpenVINO device to run inference on. AUTO lets the runtime pick based on availability.',
			},
			{
				displayName: 'Performance Hint',
				name: 'performanceHint',
				type: 'options',
				options: [
					{ name: 'Latency — minimise response time (default)', value: 'LATENCY' },
					{ name: 'Throughput — maximise requests per second',  value: 'THROUGHPUT' },
				],
				default: 'LATENCY',
			},

			// ── Advanced ──────────────────────────────────────────────────────────
			{
				displayName: 'Timeout (seconds)',
				name: 'timeout',
				type: 'number',
				default: 60,
				description: 'Maximum time to wait for a response from OVMS.',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const operation = this.getNodeParameter('operation', 0) as string;
		const credentials = await this.getCredentials('openVinoModelServerApi');
		const results: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			const transport    = ['predict', 'listModels', 'modelStatus'].includes(operation)
				? this.getNodeParameter('transport', i) as string
				: 'rest';
			const targetDevice  = this.getNodeParameter('targetDevice', i) as string;
			const perfHint      = this.getNodeParameter('performanceHint', i) as string;
			const timeoutSec    = this.getNodeParameter('timeout', i) as number;

			const restHeaders: Record<string, string> = {
				'Content-Type': 'application/json',
				'X-Target-Device': targetDevice,
				'X-Performance-Hint': perfHint,
				...(credentials.apiKey ? { Authorization: `Bearer ${credentials.apiKey}` } : {}),
			};

			const grpcMeta = { targetDevice, performanceHint: perfHint };

			let modelName: string | undefined;

			try {
				let result: unknown;

				// ── predict ────────────────────────────────────────────────────────
				if (operation === 'predict') {
					modelName = this.getNodeParameter('modelName', i) as string;
					const rawInput = this.getNodeParameter('inputData', i);
					const inputData = typeof rawInput === 'string' ? JSON.parse(rawInput) : rawInput;

					if (transport === 'grpc') {
						this.logger.info(`[openvino:predict] gRPC → ${credentials.grpcHost}:${credentials.grpcPort} model=${modelName} device=${targetDevice}`);
						const client = new OvmsGrpcClient();
						client.connect(credentials.grpcHost as string, credentials.grpcPort as number);
						try {
							result = await client.modelInfer({ model_name: modelName, inputs: (inputData as any).inputs ?? [] }, grpcMeta);
						} finally {
							client.close();
						}
					} else {
						this.logger.info(`[openvino:predict] REST → ${credentials.gatewayUrl} model=${modelName} device=${targetDevice}`);
						result = await this.helpers.httpRequest({
							method: 'POST',
							url: `${credentials.gatewayUrl}/v1/models/${modelName}:predict`,
							headers: restHeaders,
							body: inputData,
							json: true,
							timeout: timeoutSec * 1000,
						});
					}

				// ── listModels ─────────────────────────────────────────────────────
				} else if (operation === 'listModels') {
					if (transport === 'grpc') {
						const client = new OvmsGrpcClient();
						client.connect(credentials.grpcHost as string, credentials.grpcPort as number);
						try {
							result = await client.serverMetadata(grpcMeta);
						} finally {
							client.close();
						}
					} else {
						result = await this.helpers.httpRequest({
							method: 'GET',
							url: `${credentials.gatewayUrl}/v1/models`,
							headers: restHeaders,
							json: true,
							timeout: timeoutSec * 1000,
						});
					}

				// ── modelStatus ────────────────────────────────────────────────────
				} else if (operation === 'modelStatus') {
					modelName = this.getNodeParameter('modelName', i) as string;

					if (transport === 'grpc') {
						const client = new OvmsGrpcClient();
						client.connect(credentials.grpcHost as string, credentials.grpcPort as number);
						try {
							const ready = await client.modelReady(modelName, '', grpcMeta);
							result = { model: modelName, ready };
						} finally {
							client.close();
						}
					} else {
						result = await this.helpers.httpRequest({
							method: 'GET',
							url: `${credentials.gatewayUrl}/v1/models/${modelName}`,
							headers: restHeaders,
							json: true,
							timeout: timeoutSec * 1000,
						});
					}

				// ── stubs (W4-W5) ──────────────────────────────────────────────────
				} else {
					this.logger.info(`[openvino:${operation}] stub — full implementation in W4/W5`);
					result = { _stub: true, operation, note: 'documentInference lands W4, embeddings + chatCompletion land W5' };
				}

				results.push({ json: result as unknown as IDataObject, pairedItem: { item: i } });

			} catch (err) {
				const friendly = classifyOvmsError(err as Error, {
					transport,
					modelName,
					gatewayUrl: credentials.gatewayUrl as string,
					grpcHost: credentials.grpcHost as string,
					grpcPort: credentials.grpcPort as number,
				});

				if (this.continueOnFail()) {
					results.push({ json: { error: friendly }, pairedItem: { item: i } });
				} else {
					throw new NodeOperationError(this.getNode(), friendly, { itemIndex: i });
				}
			}
		}

		return [results];
	}
}
