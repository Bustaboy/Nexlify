// src/main.tsx
// NEXLIFY NEURAL INTERFACE - React Entry Point
// Where the chrome meets the meat

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/global.css'; // Create this for global styles

// Initialize error boundary for production
if (import.meta.env.PROD) {
  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    // Send to error tracking service in production
  });
}

// Disable right-click in production
if (import.meta.env.PROD) {
  document.addEventListener('contextmenu', (e) => e.preventDefault());
}

// Mount React app
const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Failed to find root element. Neural link severed.');
}

// Remove loading screen when React is ready
const removeLoader = () => {
  const loader = document.getElementById('loading-screen');
  if (loader) {
    loader.style.opacity = '0';
    setTimeout(() => loader.remove(), 500);
  }
};

// Create root and render
const root = ReactDOM.createRoot(rootElement);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Remove loader after render
setTimeout(removeLoader, 100);

// Hot Module Replacement for development
if (import.meta.hot) {
  import.meta.hot.accept();
}