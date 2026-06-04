/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
	preset: 'ts-jest',
	testEnvironment: 'node',
	roots: ['<rootDir>/tests'],
	testMatch: ['**/*.test.ts'],
	collectCoverageFrom: [
		'nodes/**/*.ts',
		'credentials/**/*.ts',
		'!nodes/**/grpc/kserve.proto',
	],
	coverageDirectory: 'coverage',
	coverageThreshold: {
		global: {
			branches: 70,
			functions: 80,
			lines: 80,
			statements: 80,
		},
	},
};
