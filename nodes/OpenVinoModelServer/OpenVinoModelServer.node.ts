import {
	IDataObject,
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

import { OvmsGrpcClient } from './grpc/client';

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
			// ── Operation ───────────────────────────────────────────────────────────
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

			// ── Transport (only for ops that can use gRPC) ───────────────────────
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

			// ── Predict params ───────────────────────────────────────────────────
			{
				displayName: 'Model Name',
				name: 'modelName',
				type: 'string',
				default: 'text-classifier',
				displayOptions: {
					show: { operation: ['predict', 'modelStatus'] },
				},
			},
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
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const operation = this.getNodeParameter('operation', 0) as string;
		const credentials = await this.getCredentials('openVinoModelServerApi');
		const results: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			try {
				let result: unknown;

				if (operation === 'predict') {
					const transport  = this.getNodeParameter('transport', i) as string;
					const modelName  = this.getNodeParameter('modelName', i) as string;
					const rawInput   = this.getNodeParameter('inputData', i);
					const inputData  = typeof rawInput === 'string' ? JSON.parse(rawInput) : rawInput;

					if (transport === 'grpc') {
						this.logger.info(`[openvino:predict] gRPC → ${credentials.grpcHost}:${credentials.grpcPort} model=${modelName}`);
						const client = new OvmsGrpcClient();
						client.connect(credentials.grpcHost as string, credentials.grpcPort as number);
						try {
							result = await client.modelInfer({
								model_name: modelName,
								inputs: (inputData as any).inputs ?? [],
							});
						} finally {
							client.close();
						}
					} else {
						this.logger.info(`[openvino:predict] REST → ${credentials.gatewayUrl} model=${modelName}`);
						result = await this.helpers.httpRequest({
							method: 'POST',
							url: `${credentials.gatewayUrl}/v1/models/${modelName}:predict`,
							headers: {
								'Content-Type': 'application/json',
								...(credentials.apiKey ? { Authorization: `Bearer ${credentials.apiKey}` } : {}),
							},
							body: inputData,
							json: true,
						});
					}

				} else if (operation === 'listModels') {
					const transport = this.getNodeParameter('transport', i) as string;

					if (transport === 'grpc') {
						const client = new OvmsGrpcClient();
						client.connect(credentials.grpcHost as string, credentials.grpcPort as number);
						try {
							result = await client.serverMetadata();
						} finally {
							client.close();
						}
					} else {
						result = await this.helpers.httpRequest({
							method: 'GET',
							url: `${credentials.gatewayUrl}/v1/models`,
							json: true,
						});
					}

				} else if (operation === 'modelStatus') {
					const transport = this.getNodeParameter('transport', i) as string;
					const modelName = this.getNodeParameter('modelName', i) as string;

					if (transport === 'grpc') {
						const client = new OvmsGrpcClient();
						client.connect(credentials.grpcHost as string, credentials.grpcPort as number);
						try {
							const ready = await client.modelReady(modelName);
							result = { model: modelName, ready };
						} finally {
							client.close();
						}
					} else {
						result = await this.helpers.httpRequest({
							method: 'GET',
							url: `${credentials.gatewayUrl}/v1/models/${modelName}`,
							json: true,
						});
					}

				} else {
					// documentInference, embeddings, chatCompletion — full impl in W2
					this.logger.info(`[openvino:${operation}] stub — full implementation in W2`);
					result = { _stub: true, operation, note: 'Full implementation lands W2' };
				}

				results.push({ json: result as unknown as IDataObject, pairedItem: { item: i } });

			} catch (err) {
				if (this.continueOnFail()) {
					results.push({ json: { error: (err as Error).message }, pairedItem: { item: i } });
				} else {
					throw new NodeOperationError(this.getNode(), err as Error, { itemIndex: i });
				}
			}
		}

		return [results];
	}
}
