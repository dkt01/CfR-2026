import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig({
  server: {
    fs: {
      allow: [repositoryRoot],
    },
  },
});
