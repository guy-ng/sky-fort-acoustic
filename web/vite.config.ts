import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'http://localhost:8000',
        ws: true,
        timeout: 0,
        configure: (proxy) => {
          const silenced = new Set(['EPIPE', 'ECONNREFUSED', 'ECONNRESET', 'ERR_STREAM_WRITE_AFTER_END'])
          const isSilenced = (err: Error) => silenced.has((err as NodeJS.ErrnoException).code ?? '')

          proxy.on('error', (err, _req, res) => {
            if (isSilenced(err)) return
            console.error('[ws proxy]', err.message)
            // Prevent unhandled crash on non-socket responses
            if (res && 'writeHead' in res && !res.headersSent) {
              (res as import('http').ServerResponse).writeHead(502).end()
            }
          })
          proxy.on('proxyReqWs', (_proxyReq, _req, socket) => {
            socket.on('error', (err) => {
              if (isSilenced(err)) return
              console.error('[ws proxy socket]', err.message)
            })
          })
        },
      },
      '/health': 'http://localhost:8000',
    },
  },
  build: {
    outDir: 'dist',
  },
})
