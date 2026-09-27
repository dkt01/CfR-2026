import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig({
  server: {
    fs: {
      allow: [repositoryRoot],
    },
  },
  build: {
    rollupOptions: {
      output: {
        // Keep meshes and textures under their own names.  gzweb finds a
        // world's model:// URIs in the asset list by FILE NAME, so a hashed
        // straw_bale_0-Bx1c.png is never matched and the bale renders with
        // no texture.  (The dev server serves them unhashed already.)
        assetFileNames: (asset) =>
          /\.(png|stl|dae|obj)$/i.test(asset.names?.[0] ?? asset.name ?? "")
            ? "assets/[name][extname]"
            : "assets/[name]-[hash][extname]",
      },
    },
  },
});
