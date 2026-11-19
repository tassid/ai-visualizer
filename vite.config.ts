/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `base` precisa ser o nome do repositório para funcionar no GitHub Pages
// (https://tassid.github.io/ai-visualizer/). Localmente: http://localhost:5173/ai-visualizer/
export default defineConfig({
  plugins: [react()],
  base: '/ai-visualizer/',
  build: { chunkSizeWarningLimit: 900 },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
