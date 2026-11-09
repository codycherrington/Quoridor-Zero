import { defineConfig } from 'vite';

export default defineConfig({
  // relative base so the build works on GitHub Pages or any sub-path
  base: './',
  worker: { format: 'es' },
});
