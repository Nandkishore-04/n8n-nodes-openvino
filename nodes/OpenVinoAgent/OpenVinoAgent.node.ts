import {
	IDataObject,
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

import { runAgent } from './loop';
import { buildTools, ChatFn, ToolContext } from './tools/builtIn';

export class OpenVinoAgent implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'OpenVINO Agent',
		name: 'openVinoAgent',
		icon: 'file:openvino.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{"agent loop"}}',
		description: 'Local agentic loop on Qwen3-4B — reasons over a document and calls tools',
		defaults: { name: 'OpenVINO Agent' },
		inputs: ['main'],
		outputs: ['main'],
		credentials: [{ name: 'openVinoModelServerApi', required: true }],
		properties: [
			{
				displayName: 'Input Field',
				name: 'inputField',
				type: 'string',
				default: 'text',
				description: 'Field on the incoming item holding the document text to reason over',
			},
			{
				displayName: 'System Prompt',
				name: 'systemPrompt',
				type: 'string',
				default: 'You analyze ANY document type (invoice, resume, proposal, receipt, contract, etc.). Steps: (1) call extract_fields ONCE with a schema fitting the type — pass ONLY the schema like {"schema":{"name":"str"}}, NEVER the document text. (2) If monetary totals exist, call validate_math. (3) Then finish. Call flag_for_review ONLY if the text is unreadable. CRITICAL: your final must be SHORT — do NOT repeat the extracted fields (they are already captured). End with exactly: {"final": {"decision": "enriched|flagged|duplicate", "document_type": "...", "summary": "one sentence"}}.',
				typeOptions: { rows: 5 },
			},
			{
				displayName: 'LLM Model',
				name: 'llmModel',
				type: 'string',
				default: 'OpenVINO/Qwen3-4B-int4-ov',
			},
			{
				displayName: 'Max Iterations',
				name: 'maxIters',
				type: 'number',
				default: 8,
				description: 'Safety cap on the agent loop. Returns the best result so far if exceeded.',
			},
			{
				displayName: 'No-Think Mode',
				name: 'noThink',
				type: 'boolean',
				default: true,
			},
			{
				displayName: 'Temperature',
				name: 'temperature',
				type: 'number',
				typeOptions: { minValue: 0, maxValue: 2, numberPrecision: 2 },
				default: 0.2,
			},
			{
				displayName: 'Per-Call Timeout (Seconds)',
				name: 'callTimeout',
				type: 'number',
				default: 300,
				description: 'Max wait per LLM turn. Raise for large documents on CPU (Qwen3 is ~7 tok/s).',
			},
			{
				displayName: 'Max Tokens',
				name: 'maxTokens',
				type: 'number',
				default: 1024,
				description: 'Cap per LLM turn. Higher avoids truncating extract_fields output; slower on CPU.',
			},
			{
				displayName: 'Tool Backends',
				name: 'toolBackends',
				type: 'collection',
				placeholder: 'Add backend',
				default: {},
				description: 'Optional endpoints that activate the external tools (retry/dedup/vendor/recall)',
				options: [
					{
						displayName: 'Check-Duplicate URL',
						name: 'checkDuplicateUrl',
						type: 'string',
						default: '',
						description: 'Workflow-as-Tool webhook for check_duplicate',
					},
					{
						displayName: 'Lookup-Vendor URL',
						name: 'lookupVendorUrl',
						type: 'string',
						default: '',
						description: 'Workflow-as-Tool webhook for lookup_vendor',
					},
					{
						displayName: 'Qdrant Collection',
						name: 'qdrantCollection',
						type: 'string',
						default: 'document_extractions',
					},
					{
						displayName: 'Qdrant URL',
						name: 'qdrantUrl',
						type: 'string',
						default: 'http://qdrant:6333',
						description: 'Qdrant base URL for recall (needs BGE embeddings on the gateway)',
					},
					{
						displayName: 'Source Binary Property',
						name: 'binaryProperty',
						type: 'string',
						default: '',
						description: 'Binary field holding the source document — enables retry_document_extraction',
					},
				],
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const credentials = await this.getCredentials('openVinoModelServerApi');
		const results: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			try {
				const inputField = this.getNodeParameter('inputField', i) as string;
				const system = this.getNodeParameter('systemPrompt', i) as string;
				const llmModel = this.getNodeParameter('llmModel', i) as string;
				const maxIters = this.getNodeParameter('maxIters', i) as number;
				const noThink = this.getNodeParameter('noThink', i) as boolean;
				const temperature = this.getNodeParameter('temperature', i) as number;
				const callTimeout = this.getNodeParameter('callTimeout', i) as number;
				const maxTokens = this.getNodeParameter('maxTokens', i) as number;
				const backends = this.getNodeParameter('toolBackends', i, {}) as IDataObject;

				const input = String((items[i].json as IDataObject)[inputField] ?? '');
				if (!input) {
					throw new NodeOperationError(this.getNode(), `Input field '${inputField}' is empty on item ${i}.`, { itemIndex: i });
				}

				// chatFn → gateway /v1/chat/completions (prompt-based; we parse tool calls in the loop)
				const chat: ChatFn = async (messages) => {
					const resp = await this.helpers.httpRequest({
						method: 'POST',
						url: `${credentials.gatewayUrl}/v1/chat/completions`,
						headers: {
							'Content-Type': 'application/json',
							...(credentials.apiKey ? { Authorization: `Bearer ${credentials.apiKey}` } : {}),
						},
						body: { model: llmModel, messages, temperature, max_tokens: maxTokens },
						json: true,
						timeout: callTimeout * 1000,
					}) as any;
					return resp?.choices?.[0]?.message?.content ?? '';
				};

				// Optional source document (base64) → enables retry_document_extraction
				let sourceFile: ToolContext['sourceFile'];
				const binaryProperty = String(backends.binaryProperty ?? '');
				if (binaryProperty && items[i].binary?.[binaryProperty]) {
					const buf = await this.helpers.getBinaryDataBuffer(i, binaryProperty);
					sourceFile = { dataB64: buf.toString('base64'), filename: items[i].binary![binaryProperty].fileName ?? 'document' };
				}

				const ctx: ToolContext = {
					chat,
					http: (opts) => this.helpers.httpRequest(opts as any),
					gatewayUrl: credentials.gatewayUrl as string,
					apiKey: credentials.apiKey as string,
					noThink,
					documentText: input,
					sourceFile,
					checkDuplicateUrl: String(backends.checkDuplicateUrl ?? '') || undefined,
					lookupVendorUrl: String(backends.lookupVendorUrl ?? '') || undefined,
					qdrant: backends.qdrantUrl
						? { url: String(backends.qdrantUrl), collection: String(backends.qdrantCollection ?? 'document_extractions') }
						: undefined,
				};

				const tools = buildTools(ctx);
				this.logger.info(`[openvino:agent] running loop (max ${maxIters} iters, ${tools.size} tools) over field '${inputField}'`);
				const result = await runAgent({ chat, tools, system, input, maxIters, noThink });

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
