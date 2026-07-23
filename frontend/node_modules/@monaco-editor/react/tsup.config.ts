import { defineConfig } from 'tsup';

export default defineConfig((opts) => ({
  clean: true,
  dts: true,
  entry: ['src/index.ts'],
  format: ['cjs', 'esm'],
  minify: !opts.watch,
  sourcemap: true,
  target: 'esnext',
  outDir: 'dist',
}));
