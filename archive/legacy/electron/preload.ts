/**
 * Nexlify Preload Script
 * Secure bridge between renderer and main process
 */

import { contextBridge, ipcRenderer } from 'electron';

// Define typed channels for security
const ALLOWED_CHANNELS = {
  invoke: [
    'system:getInfo',
    'config:get',
    'config:set',
    'fs:readFile',
    'fs:writeFile',
    'fs:listFiles',
    'dialog:showSaveDialog',
    'dialog:showOpenDialog',
    'notification:show',
    'window:minimize',
    'window:maximize',
    'window:close',
    'shell:openExternal',
    'updater:checkForUpdates'
  ],
  send: [
    'trading:statusUpdate'
  ],
  receive: [
    'command',
    'navigate'
  ]
} as const;

// Type definitions
export interface NexlifyAPI {
  system: {
    getInfo: () => Promise<SystemInfo>;
    platform: NodeJS.Platform;
  };
  config: {
    get: (key?: string) => Promise<any>;
    set: (key: string, value: any) => Promise<boolean>;
  };
  fs: {
    readFile: (path: string) => Promise<string>;
    writeFile: (path: string, content: string) => Promise<void>;
    listFiles: (path: string) => Promise<string[]>;
  };
  dialog: {
    showSaveDialog: (options: any) => Promise<Electron.SaveDialogReturnValue>;
    showOpenDialog: (options: any) => Promise<Electron.OpenDialogReturnValue>;
  };
  notification: {
    show: (options: NotificationOptions) => Promise<'clicked' | 'closed'>;
  };
  window: {
    minimize: () => Promise<void>;
    maximize: () => Promise<void>;
    close: () => Promise<void>;
  };
  shell: {
    openExternal: (url: string) => Promise<boolean>;
  };
  updater: {
    checkForUpdates: () => Promise<any>;
  };
  trading: {
    sendStatusUpdate: (status: TradingStatus) => void;
  };
  on: (channel: string, callback: (...args: any[]) => void) => void;
  off: (channel: string, callback: (...args: any[]) => void) => void;
}

interface SystemInfo {
  version: string;
  platform: NodeJS.Platform;
  arch: string;
  paths: {
    userData: string;
    logs: string;
    config: string;
  };
}

interface NotificationOptions {
  title: string;
  body: string;
  silent?: boolean;
  urgency?: 'low' | 'normal' | 'critical';
}

interface TradingStatus {
  active: boolean;
  pnl: string;
  positions: number;
  alerts: number;
}

// Create secure API
const nexlifyAPI: NexlifyAPI = {
  system: {
    getInfo: () => ipcRenderer.invoke('system:getInfo'),
    platform: process.platform
  },
  
  config: {
    get: (key?: string) => ipcRenderer.invoke('config:get', key),
    set: (key: string, value: any) => ipcRenderer.invoke('config:set', key, value)
  },
  
  fs: {
    readFile: (path: string) => ipcRenderer.invoke('fs:readFile', path),
    writeFile: (path: string, content: string) => ipcRenderer.invoke('fs:writeFile', path, content),
    listFiles: (path: string) => ipcRenderer.invoke('fs:listFiles', path)
  },
  
  dialog: {
    showSaveDialog: (options: any) => ipcRenderer.invoke('dialog:showSaveDialog', options),
    showOpenDialog: (options: any) => ipcRenderer.invoke('dialog:showOpenDialog', options)
  },
  
  notification: {
    show: (options: NotificationOptions) => ipcRenderer.invoke('notification:show', options)
  },
  
  window: {
    minimize: () => ipcRenderer.invoke('window:minimize'),
    maximize: () => ipcRenderer.invoke('window:maximize'),
    close: () => ipcRenderer.invoke('window:close')
  },
  
  shell: {
    openExternal: (url: string) => ipcRenderer.invoke('shell:openExternal', url)
  },
  
  updater: {
    checkForUpdates: () => ipcRenderer.invoke('updater:checkForUpdates')
  },
  
  trading: {
    sendStatusUpdate: (status: TradingStatus) => {
      ipcRenderer.send('trading:statusUpdate', status);
    }
  },
  
  on: (channel: string, callback: (...args: any[]) => void) => {
    if (ALLOWED_CHANNELS.receive.includes(channel as any)) {
      const subscription = (_: any, ...args: any[]) => callback(...args);
      ipcRenderer.on(channel, subscription);
    }
  },
  
  off: (channel: string, callback: (...args: any[]) => void) => {
    if (ALLOWED_CHANNELS.receive.includes(channel as any)) {
      ipcRenderer.removeListener(channel, callback);
    }
  }
};

// Expose to renderer
contextBridge.exposeInMainWorld('nexlify', nexlifyAPI);

// Add TypeScript support for window.nexlify
declare global {
  interface Window {
    nexlify: NexlifyAPI;
  }
}

// Log that preload is ready
console.log('🔒 Nexlify preload initialized - Secure bridge established');
