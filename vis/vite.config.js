import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// https://cn.vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 8080,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8108", 
        changeOrigin: true,
        // 关键：把 /api 前缀去掉
        // /api/io-collection -> /io-collection
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
