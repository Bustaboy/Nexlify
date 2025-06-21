@echo off
REM apply_fixes.bat
REM Location: C:\Nexlify\apply_fixes.bat
REM Purpose: Apply the permanent fixes for NEXLIFY

echo ==========================================
echo NEXLIFY PERMANENT FIX APPLICATION
echo ==========================================
echo.

echo Step 1: Creating backup...
set timestamp=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=%timestamp: =0%
mkdir backup_%timestamp% 2>nul

if exist "vite.config.ts" copy "vite.config.ts" "backup_%timestamp%\vite.config.ts.bak"
if exist "tsconfig.json" copy "tsconfig.json" "backup_%timestamp%\tsconfig.json.bak"
if exist "src-tauri\tauri.conf.json" copy "src-tauri\tauri.conf.json" "backup_%timestamp%\tauri.conf.json.bak"

echo.
echo Step 2: Cleaning caches...
if exist "node_modules\.vite" rmdir /s /q "node_modules\.vite"
if exist "src-tauri\target\debug\deps" rmdir /s /q "src-tauri\target\debug\deps"

echo.
echo Step 3: Creating missing directories...
if not exist "src\stores" mkdir "src\stores"
if not exist "src\components\auth" mkdir "src\components\auth"
if not exist "src\components\dashboard" mkdir "src\components\dashboard"
if not exist "src\components\status" mkdir "src\components\status"
if not exist "src\components\effects" mkdir "src\components\effects"
if not exist "src\components\ui" mkdir "src\components\ui"

echo.
echo ==========================================
echo IMPORTANT: Now you need to:
echo ==========================================
echo.
echo 1. Create/Update these configuration files:
echo    - vite.config.ts
echo    - tsconfig.json
echo    - tsconfig.node.json
echo    - src-tauri\tauri.conf.json
echo.
echo 2. Copy the content from the artifacts above
echo.
echo 3. Check if component files exist by running:
echo    verify_components.bat
echo.
echo 4. If files are missing, look in archive folders
echo    or create stub files
echo.
echo 5. Run: pnpm tauri:dev
echo.
echo ==========================================
pause