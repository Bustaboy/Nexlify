@echo off
cd /d C:\Nexlify
color 0A
cls

echo.
echo  ███╗   ██╗███████╗██╗  ██╗██╗     ██╗███████╗██╗   ██╗
echo  ████╗  ██║██╔════╝╚██╗██╔╝██║     ██║██╔════╝╚██╗ ██╔╝
echo  ██╔██╗ ██║█████╗   ╚███╔╝ ██║     ██║█████╗   ╚████╔╝ 
echo  ██║╚██╗██║██╔══╝   ██╔██╗ ██║     ██║██╔══╝    ╚██╔╝  
echo  ██║ ╚████║███████╗██╔╝ ██╗███████╗██║██║        ██║   
echo  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝        ╚═╝   
echo.
echo        ARASAKA NEURAL-NET TRADING MATRIX v2.0.7.7
echo ============================================================
echo.

echo [+] Initializing Nexlify Trading Matrix...
echo.

echo Step 1: Starting Neural-Net API Server...
start "Nexlify Neural-Net API" cmd /k "python arasaka_neural_net.py"
timeout /t 5 /nobreak >nul

echo Step 2: Starting Cyberpunk GUI Interface...
start "Nexlify Trading GUI" cmd /k "python cyber_gui.py"

echo.
echo ============================================================
echo  ✅ NEXLIFY TRADING MATRIX IS ONLINE!
echo ============================================================
echo.
echo  📌 Default PIN: 2077 (change in settings!)
echo  🔐 Enter API keys on first launch
echo  💰 Set your BTC wallet address
echo  🎮 Use KILL SWITCH for emergency stop
echo.
echo  To stop: Close both windows or use the KILL SWITCH in GUI
echo ============================================================
echo.
pause
