import { OpenVinoModelServerApi } from '../../../credentials/OpenVinoModelServerApi.credentials';

describe('OpenVinoModelServerApi credential', () => {
	const cred = new OpenVinoModelServerApi();

	it('has the expected name and display name', () => {
		expect(cred.name).toBe('openVinoModelServerApi');
		expect(cred.displayName).toBe('OpenVINO Model Server API');
	});

	it('exposes all five connection fields', () => {
		const names = cred.properties.map((p) => p.name);
		expect(names).toEqual(['gatewayUrl', 'llmServerUrl', 'grpcHost', 'grpcPort', 'apiKey']);
	});

	it('defaults the gateway and gRPC endpoints to the compose service names', () => {
		const byName = Object.fromEntries(cred.properties.map((p) => [p.name, p]));
		expect(byName.gatewayUrl.default).toBe('http://gateway:8000');
		expect(byName.grpcHost.default).toBe('ovms');
		expect(byName.grpcPort.default).toBe(9000);
	});

	it('masks the API key field', () => {
		const apiKey = cred.properties.find((p) => p.name === 'apiKey');
		expect(apiKey?.typeOptions?.password).toBe(true);
	});
});
