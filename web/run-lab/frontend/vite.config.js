import { defineConfig } from "vite";

// `npm run dev` for UI work: the Python server (run.sh) still serves /api.
export default defineConfig({
    server: {
        port: 5180,
        proxy: { "/api": "http://127.0.0.1:8765" },
    },
    build: { chunkSizeWarningLimit: 1200 },
});
