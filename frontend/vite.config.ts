import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // The production build is served by the FastAPI process itself (see
  // rag/api/app.py), so the output directory is what that mount expects.
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    // Dev-mode only: the UI and the API share one origin in production, so
    // proxying here keeps `/api/...` meaning the same thing in both.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: false,
      },
    },
  },
})
