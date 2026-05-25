import { ICredentialType, INodeProperties } from 'n8n-workflow';

export class OpenVinoModelServerApi implements ICredentialType {
	name = 'openVinoModelServerApi';
	displayName = 'OpenVINO Model Server API';
	documentationUrl = 'https://docs.openvino.ai/2026/model-server/ovms_what_is_openvino_model_server.html';

	properties: INodeProperties[] = [
		{
			displayName: 'Gateway URL',
			name: 'gatewayUrl',
			type: 'string',
			default: 'http://gateway:8000',
			placeholder: 'http://gateway:8000',
			description: 'Base URL of the Python gateway that fronts OVMS Classic (BSRGAN, PP-OCRv5, BGE).',
		},
		{
			displayName: 'LLM Server URL',
			name: 'llmServerUrl',
			type: 'string',
			default: 'http://ovms-llm:8000',
			placeholder: 'http://ovms-llm:8000',
			description: 'Base URL of the OVMS GenAI endpoint serving the chat LLM (OpenAI-compatible /v3 API).',
		},
		{
			displayName: 'gRPC Host',
			name: 'grpcHost',
			type: 'string',
			default: 'ovms',
			description: 'Hostname of the OVMS Classic gRPC endpoint (KServe v2).',
		},
		{
			displayName: 'gRPC Port',
			name: 'grpcPort',
			type: 'number',
			default: 9000,
			description: 'Port of the OVMS Classic gRPC endpoint.',
		},
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			description: 'Optional bearer token forwarded to the gateway and LLM server. Leave empty if auth is disabled.',
		},
	];
}
