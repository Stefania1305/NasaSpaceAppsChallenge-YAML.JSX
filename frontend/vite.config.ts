import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The built app is served by the Python backend at /app (see server.py), so assets live under /app/.
// In development (`npm run dev`), API calls are forwarded to the backend on port 8000.
export default defineConfig({
  base: '/app/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,     // MapLibre alone is ~1 MB
  },
});
