// src/main.tsx
// NEXLIFY NEURAL INTERFACE - React Entry Point
// Where the chrome meets the meat

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/global.css';

if (import.meta.env.VITE_NODE_ENV === 'production') {
  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
  });
}

if (import.meta.env.VITE_NODE_ENV === 'production') {
  document.addEventListener('contextmenu', (e) => e.preventDefault());
}

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Failed to find root element. Neural link severed.');
}

const removeLoader = () => {
  const loader = document.getElementById('loading-screen');
  if (loader) {
    loader.style.opacity = '0';
    setTimeout(() => loader.remove(), 500);
  }
};

const root = ReactDOM.createRoot(rootElement);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

setTimeout(removeLoader, 100);

if (import.meta.hot) {
  import.meta.hot.accept();
}
