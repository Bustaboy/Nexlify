# verify_components_fixed.ps1
# Location: C:\Nexlify\verify_components_fixed.ps1
# Purpose: Verify which components exist and which need to be created/restored

Write-Host "NEXLIFY COMPONENT VERIFICATION PROTOCOL" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor DarkGray

# Components that App.tsx is trying to import
$requiredComponents = @{
    "Stores" = @(
        "src/stores/authStore.ts",
        "src/stores/marketStore.ts",
        "src/stores/tradingStore.ts"
    )
    "Components" = @(
        "src/components/auth/LoginScreen.tsx",
        "src/components/dashboard/TradingDashboard.tsx",
        "src/components/status/SystemStatus.tsx",
        "src/components/effects/NeuralBackground.tsx",
        "src/components/ui/CyberpunkLoader.tsx"
    )
}

$missingFiles = @()
$foundFiles = @()
$possibleMatches = @{}

Write-Host ""
Write-Host "Checking Required Files:" -ForegroundColor Yellow

foreach ($category in $requiredComponents.Keys) {
    Write-Host ""
    Write-Host "  $($category):" -ForegroundColor Cyan
    
    foreach ($file in $requiredComponents[$category]) {
        if (Test-Path $file) {
            Write-Host "    [OK] $file" -ForegroundColor Green
            $foundFiles += $file
        } else {
            Write-Host "    [MISSING] $file" -ForegroundColor Red
            $missingFiles += $file
            
            # Search for similar files in archive
            $filename = Split-Path $file -Leaf
            $searchPattern = "*$filename"
            
            $matches = Get-ChildItem -Path "archive" -Filter $searchPattern -Recurse -ErrorAction SilentlyContinue
            if ($matches) {
                $possibleMatches[$file] = $matches.FullName
            }
        }
    }
}

Write-Host ""
Write-Host "VERIFICATION SUMMARY:" -ForegroundColor Yellow
Write-Host "  Found: $($foundFiles.Count) files" -ForegroundColor Green
Write-Host "  Missing: $($missingFiles.Count) files" -ForegroundColor Red

if ($possibleMatches.Count -gt 0) {
    Write-Host ""
    Write-Host "POSSIBLE MATCHES IN ARCHIVE:" -ForegroundColor Cyan
    foreach ($missing in $possibleMatches.Keys) {
        Write-Host ""
        Write-Host "  Missing: $missing" -ForegroundColor Yellow
        foreach ($match in $possibleMatches[$missing]) {
            $relativePath = $match -replace [regex]::Escape($PWD.Path), "."
            Write-Host "    -> Found in: $relativePath" -ForegroundColor Gray
        }
    }
}

# Check if files exist with wrong extensions
Write-Host ""
Write-Host "Checking for TypeScript/JavaScript mismatches:" -ForegroundColor Yellow
$jsFiles = Get-ChildItem -Path "src" -Include "*.js","*.jsx" -Recurse -ErrorAction SilentlyContinue
if ($jsFiles) {
    Write-Host "  Found JavaScript files that might need TypeScript conversion:" -ForegroundColor Red
    foreach ($jsFile in $jsFiles) {
        $relativePath = $jsFile.FullName -replace [regex]::Escape($PWD.Path), "."
        Write-Host "    - $relativePath" -ForegroundColor Gray
    }
}

# Generate recovery recommendations
Write-Host ""
Write-Host "RECOVERY RECOMMENDATIONS:" -ForegroundColor Cyan

if ($missingFiles.Count -eq 0) {
    Write-Host "  All required files exist! Issue is likely configuration." -ForegroundColor Green
} else {
    Write-Host "  1. Check archive folders for original files" -ForegroundColor White
    Write-Host "  2. Apply configuration fixes (vite.config.ts, tsconfig.json)" -ForegroundColor White
    Write-Host "  3. Create stub files for missing components" -ForegroundColor White
}

# Save report
$report = @{
    "timestamp" = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "found_files" = $foundFiles
    "missing_files" = $missingFiles
    "possible_matches" = $possibleMatches
}

$report | ConvertTo-Json -Depth 10 | Out-File "component_verification_report.json"
Write-Host ""
Write-Host "Report saved to: component_verification_report.json" -ForegroundColor Gray