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

export function classifyOvmsError(err: Error, ctx: {
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
					{ name: 'Chat Completion',    value: 'chatCompletion',    description: 'Single LLM call via Qwen3-4B',          action: 'Generate a chat completion' },
					{ name: 'Classify Document',  value: 'classifyDocument',  description: 'CLIP zero-shot triage on NPU → is this a processable document? (+confidence)', action: 'Classify a document image' },
					{ name: 'Document Inference', value: 'documentInference', description: 'PDF/image → OCR text + bounding boxes', action: 'Extract text from a document' },
					{ name: 'Embeddings',         value: 'embeddings',        description: 'Text → BGE-small-en vector',           action: 'Generate a text embedding' },
					{ name: 'Get Model Status',   value: 'modelStatus',       description: 'Per-model readiness check',            action: 'Get model status' },
					{ name: 'List Models',        value: 'listModels',        description: 'List all models served by OVMS',       action: 'List models' },
					{ name: 'Predict',            value: 'predict',           description: 'Run inference on a classic model',     action: 'Run inference on a classic model' },
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
				description: 'REST: {"text": "…"}. gRPC: KServe v2 inputs array — {"inputs":[{"name":"input_ids","datatype":"INT64","shape":[1,5],"contents":{"int64_contents":[…]}}]}.',
				displayOptions: {
					show: { operation: ['predict'] },
				},
			},

			// ── Document Inference params ─────────────────────────────────────────
			{
				displayName: 'Binary Property',
				name: 'binaryProperty',
				type: 'string',
				default: 'data',
				description: 'Name of the binary property holding the PDF/image to OCR or classify',
				displayOptions: {
					show: { operation: ['documentInference', 'classifyDocument'] },
				},
			},
			{
				displayName: 'PDF DPI',
				name: 'dpi',
				type: 'number',
				default: 200,
				description: 'Render resolution for SCANNED PDF pages that need OCR (digital PDFs use the text layer directly, ignoring this)',
				displayOptions: {
					show: { operation: ['documentInference'] },
				},
			},
			{
				displayName: 'Enhance (Super-Resolution)',
				name: 'enhance',
				type: 'boolean',
				default: false,
				description: 'Whether to run low-quality IMAGES through text-sr super-resolution before OCR. Used by the low-confidence retry pass; the result is only kept if it reads better.',
				displayOptions: {
					show: { operation: ['documentInference'] },
				},
			},

			// ── Chat Completion params ────────────────────────────────────────────
			{
				displayName: 'LLM Model',
				name: 'llmModel',
				type: 'string',
				default: 'OpenVINO/Qwen3-4B-int4-ov',
				displayOptions: { show: { operation: ['chatCompletion'] } },
			},
			{
				displayName: 'System Prompt',
				name: 'systemPrompt',
				type: 'string',
				typeOptions: { rows: 3 },
				default: '',
				description: 'Optional system instruction (e.g. "Extract the vendor, date and total as JSON")',
				displayOptions: { show: { operation: ['chatCompletion'] } },
			},
			{
				displayName: 'User Message',
				name: 'userMessage',
				type: 'string',
				typeOptions: { rows: 4 },
				default: '',
				description: 'The prompt / document text to send to the model',
				displayOptions: { show: { operation: ['chatCompletion'] } },
			},
			{
				displayName: 'No-Think Mode',
				name: 'noThink',
				type: 'boolean',
				default: true,
				description: 'Whether to append /no_think so Qwen3 answers directly instead of emitting reasoning (faster)',
				displayOptions: { show: { operation: ['chatCompletion'] } },
			},
			{
				displayName: 'Temperature',
				name: 'temperature',
				type: 'number',
				typeOptions: { minValue: 0, maxValue: 2, numberPrecision: 2 },
				default: 0.7,
				displayOptions: { show: { operation: ['chatCompletion'] } },
			},
			{
				displayName: 'Max Tokens',
				name: 'maxTokens',
				type: 'number',
				default: 512,
				displayOptions: { show: { operation: ['chatCompletion'] } },
			},
			{
				displayName: 'Return Full Response',
				name: 'returnFull',
				type: 'boolean',
				default: false,
				description: 'Whether to return the raw OpenAI-style response. Off returns just the assistant message text.',
				displayOptions: { show: { operation: ['chatCompletion'] } },
			},

			// ── AUTO plugin ───────────────────────────────────────────────────────
			{
				displayName: 'Target Device',
				name: 'targetDevice',
				type: 'options',
				options: [
					{ name: 'CPU',                         value: 'CPU' },
					{ name: 'GPU',                         value: 'GPU' },
					{ name: 'NPU',                         value: 'NPU' },
					{ name: 'AUTO (Plugin — Let OpenVINO Decide)', value: 'AUTO' },
				],
				default: 'AUTO',
				description: 'OpenVINO device to run inference on. AUTO is the AUTO plugin — it picks the best available device (and falls back to CPU).',
			},
			{
				displayName: 'Performance Hint',
				name: 'performanceHint',
				type: 'options',
				options: [
					{ name: 'Latency — Minimise Response Time (Default)', value: 'LATENCY' },
					{ name: 'Throughput — Maximise Requests per Second',  value: 'THROUGHPUT' },
				],
				default: 'LATENCY',
			},

			// ── Advanced ──────────────────────────────────────────────────────────
			{
				displayName: 'Timeout (Seconds)',
				name: 'timeout',
				type: 'number',
				default: 60,
				description: 'Maximum time to wait for a response from OVMS',
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

				// ── documentInference (OCR via gateway) ─────────────────────────────
				} else if (operation === 'documentInference') {
					const binaryProperty = this.getNodeParameter('binaryProperty', i) as string;
					const dpi = this.getNodeParameter('dpi', i) as number;
					const enhance = this.getNodeParameter('enhance', i, false) as boolean;
					const binary = this.helpers.assertBinaryData(i, binaryProperty);
					const buffer = await this.helpers.getBinaryDataBuffer(i, binaryProperty);

					this.logger.info(`[openvino:documentInference] ${binary.fileName ?? 'file'}${enhance ? ' (enhance)' : ''} → ${credentials.gatewayUrl}/v1/document/infer`);
					result = await this.helpers.httpRequest({
						method: 'POST',
						url: `${credentials.gatewayUrl}/v1/document/infer`,
						headers: restHeaders,
						body: {
							data: buffer.toString('base64'),
							filename: binary.fileName ?? '',
							dpi,
							enhance,
							device: targetDevice,
						},
						json: true,
						timeout: timeoutSec * 1000,
					});

				// ── classifyDocument (CLIP zero-shot triage on NPU via gateway) ──────
				} else if (operation === 'classifyDocument') {
					const binaryProperty = this.getNodeParameter('binaryProperty', i) as string;
					const binary = this.helpers.assertBinaryData(i, binaryProperty);
					const buffer = await this.helpers.getBinaryDataBuffer(i, binaryProperty);

					this.logger.info(`[openvino:classifyDocument] ${binary.fileName ?? 'file'} → ${credentials.gatewayUrl}/v1/document/infer (classify)`);
					result = await this.helpers.httpRequest({
						method: 'POST',
						url: `${credentials.gatewayUrl}/v1/document/infer`,
						headers: restHeaders,
						body: {
							data: buffer.toString('base64'),
							filename: binary.fileName ?? '',
							mode: 'classify',
							device: targetDevice,
						},
						json: true,
						timeout: timeoutSec * 1000,
					});

				// ── chatCompletion (LLM via gateway → OVMS-LLM) ──────────────────────
				} else if (operation === 'chatCompletion') {
					const llmModel = this.getNodeParameter('llmModel', i) as string;
					const systemPrompt = this.getNodeParameter('systemPrompt', i) as string;
					let userMessage = this.getNodeParameter('userMessage', i) as string;
					const noThink = this.getNodeParameter('noThink', i) as boolean;
					const temperature = this.getNodeParameter('temperature', i) as number;
					const maxTokens = this.getNodeParameter('maxTokens', i) as number;
					const returnFull = this.getNodeParameter('returnFull', i) as boolean;

					if (noThink) userMessage = `${userMessage} /no_think`;
					const messages: Array<{ role: string; content: string }> = [];
					if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
					messages.push({ role: 'user', content: userMessage });

					this.logger.info(`[openvino:chatCompletion] ${llmModel} → ${credentials.gatewayUrl}/v1/chat/completions`);
					const resp = await this.helpers.httpRequest({
						method: 'POST',
						url: `${credentials.gatewayUrl}/v1/chat/completions`,
						headers: restHeaders,
						body: { model: llmModel, messages, temperature, max_tokens: maxTokens },
						json: true,
						timeout: timeoutSec * 1000,
					}) as any;

					if (returnFull) {
						result = resp;
					} else {
						let content = resp?.choices?.[0]?.message?.content ?? '';
						// strip a leading <think>…</think> block if the model still emitted one
						content = content.replace(/^\s*<think>[\s\S]*?<\/think>\s*/, '').trim();
						result = { content, usage: resp?.usage, model: resp?.model };
					}

				// ── stubs ──────────────────────────────────────────────────────────
				} else {
					this.logger.info(`[openvino:${operation}] stub — embeddings lands W5`);
					result = { _stub: true, operation, note: 'embeddings lands W5' };
				}

				// carry the input binary through so a later node (e.g. OCR after Classify) still has the file
				results.push({ json: result as unknown as IDataObject, binary: items[i].binary, pairedItem: { item: i } });

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
