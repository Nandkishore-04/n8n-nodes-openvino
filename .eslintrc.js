module.exports = {
	root: true,
	parser: '@typescript-eslint/parser',
	parserOptions: {
		ecmaVersion: 2020,
		sourceType: 'module',
	},
	plugins: ['@typescript-eslint', 'n8n-nodes-base'],
	ignorePatterns: ['dist/**', 'node_modules/**', 'coverage/**', 'tests/**', '*.js'],
	overrides: [
		{
			files: ['nodes/**/*.ts'],
			extends: ['plugin:n8n-nodes-base/nodes'],
		},
		{
			files: ['credentials/**/*.ts'],
			extends: ['plugin:n8n-nodes-base/credentials'],
			rules: {
				// Conflicts with cred-class-field-documentation-url-not-http-url: the miscased
				// rule tries to camelCase a full HTTPS URL, which breaks it. Full URLs are valid here.
				'n8n-nodes-base/cred-class-field-documentation-url-miscased': 'off',
			},
		},
	],
};
