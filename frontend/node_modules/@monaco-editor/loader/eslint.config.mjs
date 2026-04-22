import js from '@eslint/js';

export default [
  js.configs.recommended,
  {
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',
      globals: {
        // Browser globals
        window: 'readonly',
        document: 'readonly',
        navigator: 'readonly',
        console: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        fetch: 'readonly',
        Promise: 'readonly',
        // ES2020 globals
        globalThis: 'readonly',
      },
    },
    rules: {},
  },
  {
    ignores: ['**/spec.js', '**/*.spec.js', 'node_modules/**', 'lib/**'],
  },
];

