/**
 * Nexlify Electron Main Process
 * Handles system integration, security, and IPC
 */

import { app, BrowserWindow, ipcMain, Tray, Menu, Notification, shell, dialog, nativeImage } from 'electron';
import { autoUpdater } from 'electron-updater';
import path from 'path';
import fs from 'fs-extra';
import isDev from 'electron-is-dev';
import { spawn } from 'child_process';
import Store from 'electron-store';
import log from 'electron-log';
import windowStateKeeper from 'electron-window-state';

// Security imports
import helmet from 'helmet';
import { URL } from 'url';

// Configure logging
log.transports.file.level = 'info';
log.transports.file.file = path.join(app.getPath('userData'), 'logs', 'main.log');

// Secure store for sensitive data
const store = new Store({
  encryptionKey: process.env.ELECTRON_STORE_KEY || 'nexlify-default-key-change-in-production',
  schema: {
    pin: { type: 'string' },
    apiEndpoint: { type: 'string', default: 'http://localhost:8000' },
    theme: { type: 'string', default: 'cyberpunk' },
    autoStart: { type: 'boolean', default: false },
    minimizeToTray: { type: 'boolean', default: true },
    notifications: { type: 'boolean', default: true }
  }
});

// Global references
let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let apiProcess: any = null;
let isQuitting = false;

// Security: Prevent multiple instances
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    // Someone tried to run a second instance, focus our window instead
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

// Security configurations
app.commandLine.appendSwitch('disable-http-cache');
app.commandLine.appendSwitch('disable-http2');

// Paths
const RESOURCES_PATH = app.isPackaged
  ? path.join(process.resourcesPath, 'assets')
  : path.join(__dirname, '../assets');

const getAssetPath = (...paths: string[]): string => {
  return path.join(RESOURCES_PATH, ...paths);
};

// Create config directory
const configPath = path.join(app.getPath('userData'), 'config');
const logsPath = path.join(app.getPath('userData'), 'logs');
fs.ensureDirSync(configPath);
fs.ensureDirSync(logsPath);

/**
 * Start the Python API backend
 */
async function startAPIServer(): Promise<void> {
  if (apiProcess) return;

  const pythonPath = isDev ? 'python' : path.join(process.resourcesPath, 'python', 'python.exe');
  const apiScript = isDev 
    ? path.join(__dirname, '../../backend/nexlify_api.py')
    : path.join(process.resourcesPath, 'backend', 'nexlify_api.py');

  log.info('Starting API server...', { pythonPath, apiScript });

  try {
    apiProcess = spawn(pythonPath, [apiScript], {
      env: {
        ...process.env,
        ELECTRON_RUN_AS_NODE: '1',
        NEXLIFY_CONFIG_PATH: configPath,
        NEXLIFY_LOG_PATH: logsPath
      }
    });

    apiProcess.stdout.on('data', (data: Buffer) => {
      log.info(`API: ${data.toString()}`);
    });

    apiProcess.stderr.on('data', (data: Buffer) => {
      log.error(`API Error: ${data.toString()}`);
    });

    apiProcess.on('close', (code: number) => {
      log.info(`API process exited with code ${code}`);
      apiProcess = null;
      
      // Restart if not quitting
      if (!isQuitting && code !== 0) {
        setTimeout(startAPIServer, 5000);
      }
    });

    // Wait for API to be ready
    await waitForAPI();
    
  } catch (error) {
    log.error('Failed to start API server:', error);
    throw error;
  }
}

/**
 * Wait for API to respond
 */
async function waitForAPI(maxAttempts = 30): Promise<void> {
  const apiUrl = store.get('apiEndpoint') as string;
  
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const response = await fetch(`${apiUrl}/health`);
      if (response.ok) {
        log.info('API server is ready');
        return;
      }
    } catch (error) {
      // API not ready yet
    }
    
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  throw new Error('API server failed to start');
}

/**
 * Create the main application window
 */
async function createWindow(): Promise<void> {
  // Load window state
  const mainWindowState = windowStateKeeper({
    defaultWidth: 1400,
    defaultHeight: 900
  });

  // Create the browser window
  mainWindow = new BrowserWindow({
    x: mainWindowState.x,
    y: mainWindowState.y,
    width: mainWindowState.width,
    height: mainWindowState.height,
    minWidth: 1200,
    minHeight: 800,
    frame: false, // Frameless for cyberpunk aesthetic
    backgroundColor: '#0a0a0a',
    icon: getAssetPath('icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: !isDev,
      preload: path.join(__dirname, 'preload.js')
    },
    show: false,
    titleBarStyle: 'hidden',
    titleBarOverlay: {
      color: '#0a0a0a',
      symbolColor: '#00ffff'
    }
  });

  // Let window state manager handle it
  mainWindowState.manage(mainWindow);

  // Load the app
  if (isDev) {
    mainWindow.loadURL('http://localhost:3000');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadURL(`file://${path.join(__dirname, '../build/index.html')}`);
  }

  // Security: Prevent new window creation
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    // Open URLs in default browser
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Security: Validate navigation
  mainWindow.webContents.on('will-navigate', (event, url) => {
    const parsedUrl = new URL(url);
    
    if (isDev && parsedUrl.hostname === 'localhost') {
      return; // Allow in dev
    }
    
    if (!isDev && parsedUrl.protocol === 'file:') {
      return; // Allow file protocol in production
    }
    
    // Prevent navigation to external URLs
    event.preventDefault();
  });

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
    
    // Cyberpunk fade-in effect
    mainWindow?.webContents.insertCSS(`
      body {
        animation: cyberpunk-fadein 0.5s ease-in;
      }
      
      @keyframes cyberpunk-fadein {
        from {
          opacity: 0;
          filter: blur(10px) hue-rotate(180deg);
        }
        to {
          opacity: 1;
          filter: blur(0px) hue-rotate(0deg);
        }
      }
    `);
  });

  // Handle window close
  mainWindow.on('close', (event) => {
    if (!isQuitting && store.get('minimizeToTray')) {
      event.preventDefault();
      mainWindow?.hide();
      
      if (process.platform === 'darwin') {
        app.dock.hide();
      }
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

/**
 * Create system tray
 */
function createTray(): void {
  const trayIcon = nativeImage.createFromPath(getAssetPath('tray-icon.png'));
  tray = new Tray(trayIcon);
  
  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Show Nexlify',
      click: () => {
        mainWindow?.show();
        if (process.platform === 'darwin') {
          app.dock.show();
        }
      }
    },
    {
      label: 'Trading Status',
      enabled: false,
      id: 'status'
    },
    { type: 'separator' },
    {
      label: 'Start Trading',
      id: 'start-trading',
      click: () => {
        mainWindow?.webContents.send('command', 'start-trading');
      }
    },
    {
      label: 'Stop Trading',
      id: 'stop-trading',
      click: () => {
        mainWindow?.webContents.send('command', 'stop-trading');
      }
    },
    { type: 'separator' },
    {
      label: 'Settings',
      click: () => {
        mainWindow?.show();
        mainWindow?.webContents.send('navigate', '/settings');
      }
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        isQuitting = true;
        app.quit();
      }
    }
  ]);
  
  tray.setToolTip('Nexlify Trading Matrix');
  tray.setContextMenu(contextMenu);
  
  // Double click to show window
  tray.on('double-click', () => {
    mainWindow?.show();
    if (process.platform === 'darwin') {
      app.dock.show();
    }
  });
}

/**
 * Setup IPC handlers for secure communication
 */
function setupIPCHandlers(): void {
  // System info
  ipcMain.handle('system:getInfo', async () => {
    return {
      version: app.getVersion(),
      platform: process.platform,
      arch: process.arch,
      paths: {
        userData: app.getPath('userData'),
        logs: logsPath,
        config: configPath
      }
    };
  });

  // Config management
  ipcMain.handle('config:get', async (_, key?: string) => {
    return key ? store.get(key) : store.store;
  });

  ipcMain.handle('config:set', async (_, key: string, value: any) => {
    store.set(key, value);
    return true;
  });

  // File system operations
  ipcMain.handle('fs:readFile', async (_, filePath: string) => {
    // Security: Only allow reading from userData
    const safePath = path.join(app.getPath('userData'), filePath);
    
    if (!safePath.startsWith(app.getPath('userData'))) {
      throw new Error('Access denied');
    }
    
    return fs.readFile(safePath, 'utf-8');
  });

  ipcMain.handle('fs:writeFile', async (_, filePath: string, content: string) => {
    // Security: Only allow writing to userData
    const safePath = path.join(app.getPath('userData'), filePath);
    
    if (!safePath.startsWith(app.getPath('userData'))) {
      throw new Error('Access denied');
    }
    
    await fs.ensureDir(path.dirname(safePath));
    return fs.writeFile(safePath, content, 'utf-8');
  });

  ipcMain.handle('fs:listFiles', async (_, dirPath: string) => {
    const safePath = path.join(app.getPath('userData'), dirPath);
    
    if (!safePath.startsWith(app.getPath('userData'))) {
      throw new Error('Access denied');
    }
    
    return fs.readdir(safePath);
  });

  // Dialog operations
  ipcMain.handle('dialog:showSaveDialog', async (_, options) => {
    if (!mainWindow) throw new Error('No window available');
    return dialog.showSaveDialog(mainWindow, options);
  });

  ipcMain.handle('dialog:showOpenDialog', async (_, options) => {
    if (!mainWindow) throw new Error('No window available');
    return dialog.showOpenDialog(mainWindow, options);
  });

  // Notifications
  ipcMain.handle('notification:show', async (_, options: Electron.NotificationConstructorOptions) => {
    if (store.get('notifications')) {
      const notification = new Notification({
        icon: getAssetPath('icon.png'),
        ...options
      });
      
      notification.show();
      
      return new Promise((resolve) => {
        notification.on('click', () => resolve('clicked'));
        notification.on('close', () => resolve('closed'));
      });
    }
  });

  // Window controls
  ipcMain.handle('window:minimize', () => {
    mainWindow?.minimize();
  });

  ipcMain.handle('window:maximize', () => {
    if (mainWindow?.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow?.maximize();
    }
  });

  ipcMain.handle('window:close', () => {
    mainWindow?.close();
  });

  // Trading status updates from renderer
  ipcMain.on('trading:statusUpdate', (_, status: any) => {
    // Update tray menu
    const menu = tray?.getContextMenu();
    const statusItem = menu?.getMenuItemById('status');
    
    if (statusItem) {
      statusItem.label = `Status: ${status.active ? 'Trading' : 'Idle'} | P&L: ${status.pnl}`;
    }
    
    tray?.setContextMenu(menu!);
    
    // Update tray icon based on status
    if (status.active) {
      const activeIcon = nativeImage.createFromPath(getAssetPath('tray-icon-active.png'));
      tray?.setImage(activeIcon);
    } else {
      const idleIcon = nativeImage.createFromPath(getAssetPath('tray-icon.png'));
      tray?.setImage(idleIcon);
    }
  });

  // Open external links
  ipcMain.handle('shell:openExternal', async (_, url: string) => {
    // Validate URL
    try {
      const parsed = new URL(url);
      if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
        return shell.openExternal(url);
      }
    } catch (error) {
      log.error('Invalid URL:', url);
    }
    return false;
  });

  // Auto-updater
  ipcMain.handle('updater:checkForUpdates', async () => {
    return autoUpdater.checkForUpdatesAndNotify();
  });
}

/**
 * App event handlers
 */
app.whenReady().then(async () => {
  log.info('Nexlify starting up...');
  
  try {
    // Start API server first
    await startAPIServer();
    
    // Create UI
    await createWindow();
    createTray();
    setupIPCHandlers();
    
    // Auto-updater
    autoUpdater.checkForUpdatesAndNotify();
    
    log.info('Nexlify ready');
    
  } catch (error) {
    log.error('Startup failed:', error);
    dialog.showErrorBox('Startup Error', 'Failed to start Nexlify. Check logs for details.');
    app.quit();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('before-quit', () => {
  isQuitting = true;
  
  // Stop API server
  if (apiProcess) {
    apiProcess.kill();
  }
});

// Handle certificate errors
app.on('certificate-error', (event, webContents, url, error, certificate, callback) => {
  if (isDev) {
    // Ignore certificate errors in development
    event.preventDefault();
    callback(true);
  } else {
    // Use default behavior in production
    callback(false);
  }
});

// Security: Prevent DNS prefetch
app.on('web-contents-created', (_, contents) => {
  contents.on('will-navigate', (event, url) => {
    log.info('Navigation:', url);
  });
});

// Logging
process.on('uncaughtException', (error) => {
  log.error('Uncaught exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  log.error('Unhandled rejection:', reason);
});
