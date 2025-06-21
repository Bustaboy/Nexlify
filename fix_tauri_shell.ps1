# fix_tauri_shell.ps1
# Location: C:\Nexlify\fix_tauri_shell.ps1
# Purpose: Emergency hotfix for that fragged shell plugin

Write-Host "=== TAURI SHELL PLUGIN EMERGENCY SURGERY ===" -ForegroundColor Red
Write-Host "This is gonna hurt, but it'll get you back online" -ForegroundColor DarkRed
Write-Host ""

# First, backup the patient
$tauriConfig = "src-tauri/tauri.conf.json"
$backupName = "src-tauri/tauri.conf.json.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

if (Test-Path $tauriConfig) {
    Copy-Item $tauriConfig $backupName
    Write-Host "[BACKUP] Original config stashed at: $backupName" -ForegroundColor DarkGray
    
    # Load current config
    $config = Get-Content $tauriConfig -Raw | ConvertFrom-Json
    
    # Fix the shell plugin - this is where the magic happens
    Write-Host "[SURGERY] Replacing corrupted shell plugin config..." -ForegroundColor Yellow
    
    # Remove the broken shell config
    if ($config.plugins.PSObject.Properties.Name -contains "shell") {
        $config.plugins.PSObject.Properties.Remove("shell")
    }
    
    # Add the correct shell config
    $shellConfig = @{
        open = $true
        scope = @(
            @{
                name = "open"
                cmd = "cmd"
                args = @("/C", "start", @{ validator = "\\S+" })
            }
        )
    }
    
    # Add shell back with proper structure
    $config.plugins | Add-Member -NotePropertyName "shell" -NotePropertyValue $shellConfig -Force
    
    # Save the fixed config
    $config | ConvertTo-Json -Depth 10 | Out-File $tauriConfig -Encoding UTF8
    
    Write-Host "[SUCCESS] Shell plugin config repaired" -ForegroundColor Green
    Write-Host ""
    Write-Host "The ICE has been neutralized. Your neural pathways should be clear." -ForegroundColor Cyan
} else {
    Write-Host "[ERROR] Can't find tauri.conf.json - we're flying blind here!" -ForegroundColor Red
}