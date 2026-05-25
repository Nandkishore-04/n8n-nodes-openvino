import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';

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
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				options: [
					{ name: 'Predict', value: 'predict', description: 'Run a classic-model inference' },
					{ name: 'Document Inference', value: 'documentInference', description: 'PDF/image → OCR text + boxes' },
					{ name: 'Embeddings', value: 'embeddings', description: 'BGE-small-en text embeddings' },
					{ name: 'Chat Completion', value: 'chatCompletion', description: 'Single LLM call (Qwen3-4B)' },
					{ name: 'List Models', value: 'listModels', description: 'List models served by OVMS' },
					{ name: 'Get Model Status', value: 'modelStatus', description: 'Per-model readiness' },
				],
				default: 'predict',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const operation = this.getNodeParameter('operation', 0) as string;
		this.logger.info(`[openvino:${operation}] stub node — not yet implemented`);
		return [items.map((item) => ({
			json: { ...item.json, _stub: true, operation, note: 'OpenVINO node stub — implementation lands W1' },
			pairedItem: { item: 0 },
		}))];
	}
}
