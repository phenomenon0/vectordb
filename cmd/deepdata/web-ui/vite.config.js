import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/health': 'http://localhost:8080',
      '/api': 'http://localhost:8080',
      '/query': 'http://localhost:8080',
      '/scroll': 'http://localhost:8080',
      '/insert': 'http://localhost:8080',
      '/delete': 'http://localhost:8080',
      '/export': 'http://localhost:8080',
      '/import': 'http://localhost:8080',
      '/compact': 'http://localhost:8080',
      '/integrity': 'http://localhost:8080',
      '/vault': 'http://localhost:8080',
      '/v2': 'http://localhost:8080',
      '/v3': 'http://localhost:8080',
      '/admin': 'http://localhost:8080',
      '/metrics': 'http://localhost:8080',
    },
  },
})
