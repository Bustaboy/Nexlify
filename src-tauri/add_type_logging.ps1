$mainRsPath = "C:\nexlify\src-tauri\src\main.rs"
$backupPath = "C:\nexlify\src-tauri\src\main.rs.bak"
$logPath = "C:\nexlify\src-tauri\build_log.txt"

# Backup main.rs
if (Test-Path $mainRsPath) {
    Copy-Item $mainRsPath $backupPath -Force
    Write-Host "[INFO] Backed up main.rs to $backupPath" -ForegroundColor Green
} else {
    Write-Host "[ERROR] main.rs not found at $mainRsPath" -ForegroundColor Red
    exit 1
}

# Read main.rs content
$mainRsContent = Get-Content $mainRsPath -Raw -Encoding UTF8

# Define replacement snippets using here-strings
$marketCacheDeclaration = @"
let market_cache = Arc::new(MarketCache::new());
println!("market_cache type after declaration: {:?}", std::any::type_name_of_val(&market_cache));
"@

$beforeManageSnippet = @"
println!("market_cache type before manage: {:?}", std::any::type_name_of_val(&market_cache));
// Fix trading_engine declaration
let trading_engine = Arc::new(RwLock::new(TradingEngine::new()));
// Performance monitoring - keeping tabs on our chrome
"@

$beforeSetupSnippet = @"
info!("🔌 Jacking into the market matrix...");
println!("market_cache type before setup: {:?}", std::any::type_name_of_val(&market_cache));
"@

$closureSnippet = @"
println!("market_cache type in closure: {:?}", std::any::type_name_of_val(&market_cache));
async_runtime::spawn(async move {
    println!("market_cache type before initialize_market_streams: {:?}", std::any::type_name_of_val(&market_cache));
"@

# Apply replacements
$newContent = $mainRsContent -replace `
    'let market_cache = Arc::new\(MarketCache::new\(\)\);', `
    $marketCacheDeclaration -replace `
    '// Performance monitoring - keeping tabs on our chrome', `
    $beforeManageSnippet -replace `
    'info!\("🔌 Jacking into the market matrix..."\);', `
    $beforeSetupSnippet -replace `
    'async_runtime::spawn\(async move {', `
    $closureSnippet

# Write updated content
Set-Content -Path $mainRsPath -Value $newContent -Encoding UTF8
Write-Host "[INFO] Added type logging to main.rs" -ForegroundColor Green

# Build and test
Write-Host "[INFO] Running cargo build..." -ForegroundColor Yellow
cargo build 2>&1 | Tee-Object -FilePath $logPath
Write-Host "[INFO] Running pnpm tauri:dev..." -ForegroundColor Yellow
pnpm tauri:dev 2>&1 | Tee-Object -FilePath $logPath -Append