import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    // The checked-in OSM teaching graph is intentionally a separate, highly
    // compressible data chunk (about 450 KB over the wire).
    chunkSizeWarningLimit: 2500,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("maplibre-gl")) return "maplibre";
          if (id.includes("san-francisco.graph.json")) return "san-francisco-graph";
          if (id.includes("node_modules/react")) return "react";
        }
      }
    }
  }
});
