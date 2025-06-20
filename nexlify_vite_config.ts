// vite.config.ts
// NEXLIFY BUILD CONFIGURATION - Optimized for Tauri + React
// No placeholders, just pure performance chrome

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// Tauri expects a fixed port during development
const TAURI_DEV_HOST = process.env.TAURI_DEV_HOST || 'localhost';

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }) => {
  const isProduction = mode === 'production';
  const isTauri = process.env.TAURI_PLATFORM !== undefined;

  return {
    plugins: [
      react({
        // Enable Fast Refresh
        fastRefresh: !isProduction,
        // Babel config for emotion/styled components if needed
        babel: {
          plugins: [
            // Add babel plugins here if using emotion, styled-components, etc
          ],
        },
      }),
    ],

    // Prevent vite from obscuring rust errors
    clearScreen: false,

    // Development server configuration
    server: {
      port: 5173,
      strictPort: true,
      host: TAURI_DEV_HOST === 'localhost' ? false : true,
      hmr: {
        protocol: 'ws',
        host: TAURI_DEV_HOST,
        port: 5173,
      },
      watch: {
        // Tell Vite to ignore watching `src-tauri`
        ignored: ['**/src-tauri/**'],
      },
    },

    // Build configuration
    build: {
      // Tauri supports es2021
      target: ['es2021', 'chrome100', 'safari13'],
      
      // Don't minify for debug builds
      minify: isProduction ? 'esbuild' : false,
      
      // Produce sourcemaps for debug builds
      sourcemap: !isProduction,
      
      // Output directory
      outDir: 'dist',
      
      // Empty outDir on build
      emptyOutDir: true,
      
      // Chunk size warnings
      chunkSizeWarningLimit: 2000,
      
      // Rollup options
      rollupOptions: {
        input: {
          main: resolve(__dirname, 'index.html'),
        },
        output: {
          // Chunk strategy for better caching
          manualChunks: {
            'react-vendor': ['react', 'react-dom'],
            'state-vendor': ['zustand', 'immer'],
            'ui-vendor': ['framer-motion', 'react-hot-toast'],
            'chart-vendor': ['recharts', 'd3-scale', 'd3-shape'],
          },
          // Asset naming
          assetFileNames: (assetInfo) => {
            let extType = assetInfo.name?.split('.').at(-1) || 'asset';
            if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(extType)) {
              extType = 'img';
            }
            return `assets/${extType}/[name]-[hash][extname]`;
          },
          // Chunk naming
          chunkFileNames: isProduction
            ? 'assets/js/[name]-[hash].js'
            : 'assets/js/[name].js',
          // Entry naming
          entryFileNames: isProduction
            ? 'assets/js/[name]-[hash].js'
            : 'assets/js/[name].js',
        },
      },
      
      // Build optimizations
      cssCodeSplit: true,
      assetsInlineLimit: 4096, // 4kb
      
      // Terser options for production
      terserOptions: isProduction
        ? {
            compress: {
              drop_console: true,
              drop_debugger: true,
              pure_funcs: ['console.log', 'console.info'],
            },
            mangle: {
              safari10: true,
            },
            format: {
              comments: false,
            },
          }
        : undefined,
    },

    // Path resolution
    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
        '@components': resolve(__dirname, './src/components'),
        '@hooks': resolve(__dirname, './src/hooks'),
        '@stores': resolve(__dirname, './src/stores'),
        '@lib': resolve(__dirname, './src/lib'),
        '@types': resolve(__dirname, './src/types'),
        '@utils': resolve(__dirname, './src/utils'),
        '@assets': resolve(__dirname, './src/assets'),
      },
    },

    // Environment variables
    envPrefix: ['VITE_', 'TAURI_'],

    // CSS configuration
    css: {
      devSourcemap: !isProduction,
      modules: {
        localsConvention: 'camelCaseOnly',
      },
      preprocessorOptions: {
        scss: {
          // Add global SCSS variables/mixins if using SCSS
          additionalData: ``,
        },
      },
    },

    // Optimizations
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        '@tauri-apps/api',
        'zustand',
        'immer',
        'framer-motion',
      ],
      exclude: ['@tauri-apps/api'],
    },

    // Worker configuration
    worker: {
      format: 'es',
      plugins: [],
    },

    // Preview configuration (for testing production builds)
    preview: {
      port: 4173,
      strictPort: true,
    },
  };
});