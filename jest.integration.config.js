/** @type {import('ts-jest').JestConfigWithTsJest} */
// Integration tests — require the live Podman stack (run: deployment/podman-compose up -d).
// Run with: npm run test:integration
module.exports = {
	preset: 'ts-jest',
	testEnvironment: 'node',
	roots: ['<rootDir>/tests/integration'],
	testMatch: ['**/*.test.ts'],
	testTimeout: 30000,
};
