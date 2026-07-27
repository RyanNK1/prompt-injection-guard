import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Standard Vite + React config. No proxy needed here -- the app calls
// the FastAPI backend's full URL directly (see src/App.jsx), and CORS
// is already handled on the backend side (see api/main.py).
export default defineConfig({
  plugins: [react()],
})
