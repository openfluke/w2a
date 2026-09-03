import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5177, strictPort: true },
  preview: { host: "127.0.0.1", port: 4178, strictPort: true },
  optimizeDeps: {
    exclude: ["@openfluke/welvet"],
  },
  // Package re-exports Node loader; stub it so Vite does not pull fs/path.
  resolve: {
    alias: {
      "@openfluke/welvet/dist/loader.js": path.join(root, "src/loader-browser-stub.ts"),
    },
  },
});
