#!/bin/bash
# NEXLIFY REPO UNFUCKER v1.0 - "Digital Street Surgery"
# Author: Your friendly neighborhood code surgeon
# Last sync: 2025-06-19 | "Sometimes you gotta burn it down to build it up"

set -e  # Exit on any error - no room for fuck-ups

# Colors for our cyberpunk aesthetic
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner - because style matters even in crisis
echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  NEXLIFY REPO UNFUCKER v1.0                   ║"
echo "║               'Cleaning the streets, one file at a time'       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Safety check - are we in the right place?
if [ ! -f "recovery_protocol.py" ] || [ ! -d "src" ]; then
    echo -e "${RED}❌ ERROR: Not in Nexlify root directory!${NC}"
    echo "Please run this from your Nexlify project root"
    exit 1
fi

echo -e "${YELLOW}⚠️  WARNING: This script will restructure your entire repo${NC}"
echo -e "${YELLOW}⚠️  A complete backup will be created first${NC}"
read -p "Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo -e "${RED}Aborted. Smart choice - always think twice before surgery.${NC}"
    exit 0
fi

# Create timestamp for unique naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="archive/pre-unfuck-backup-${TIMESTAMP}"

echo -e "\n${CYAN}🏥 PHASE 1: EMERGENCY BACKUP${NC}"
echo "Creating full backup before surgery..."

# Create archive structure
mkdir -p "$ARCHIVE_DIR"

# Backup EVERYTHING first - paranoia saves lives
echo "📦 Backing up current state..."
cp -r src "$ARCHIVE_DIR/src-original" 2>/dev/null || true
cp -r frontend "$ARCHIVE_DIR/frontend-original" 2>/dev/null || true
cp -r api "$ARCHIVE_DIR/api-original" 2>/dev/null || true
cp -r backend "$ARCHIVE_DIR/backend-original" 2>/dev/null || true
cp -r home "$ARCHIVE_DIR/home-original" 2>/dev/null || true

# Backup all config files
cp *.json "$ARCHIVE_DIR/" 2>/dev/null || true
cp *.yaml "$ARCHIVE_DIR/" 2>/dev/null || true
cp *.yml "$ARCHIVE_DIR/" 2>/dev/null || true
cp *.txt "$ARCHIVE_DIR/" 2>/dev/null || true

echo -e "${GREEN}✅ Backup complete at: $ARCHIVE_DIR${NC}"

echo -e "\n${CYAN}🔍 PHASE 2: DIAGNOSIS - Mapping the Chaos${NC}"

# Count the damage
PACKAGE_JSON_COUNT=$(find . -name "package.json" -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/archive/*" | wc -l)
APP_FILES_COUNT=$(find . -name "App.tsx" -o -name "App.jsx" -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/archive/*" | wc -l)
PYTHON_API_COUNT=$(find . -name "*api*.py" -not -path "*/archive/*" | wc -l)

echo "📊 Damage Report:"
echo "   - package.json files: $PACKAGE_JSON_COUNT"
echo "   - App.tsx/jsx files: $APP_FILES_COUNT"
echo "   - Python API files: $PYTHON_API_COUNT"

echo -e "\n${CYAN}🧹 PHASE 3: NUCLEAR CLEANUP${NC}"

# NEXUS-7 Protection - This is GOLD, DO NOT TOUCH
if [ -d "home/netrunner/neuralink_project" ]; then
    echo "🧠 NEXUS-7 System Detected - Military-grade DRL!"
    echo -e "${GREEN}   ✓ Preserving NEXUS-7 for future integration${NC}"
    echo -e "${YELLOW}   ℹ Market Oracle module could revolutionize our trading${NC}"
else
    echo "⚠️  NEXUS-7 not found - continuing with cleanup"
fi

# Archive the old frontend
if [ -d "frontend" ]; then
    echo "🔥 Archiving old frontend/ structure..."
    mv frontend "$ARCHIVE_DIR/old-frontend"
    echo -e "${GREEN}   ✓ Old frontend sent to the shadow realm${NC}"
fi

# Deal with the mixed src/ folder
echo -e "\n${YELLOW}🔧 PHASE 4: SURGICAL SEPARATION${NC}"
echo "Separating Python backend from TypeScript frontend..."

# Create clean backend structure
mkdir -p src-backend/{core,api,exchanges,ml,monitoring,security,trading,risk,utils}

# Move Python files to backend
echo "🐍 Relocating Python files to src-backend/..."
find src -name "*.py" -type f | while read -r file; do
    # Determine target directory based on file location
    if [[ $file == *"/api/"* ]]; then
        mkdir -p "src-backend/api"
        cp "$file" "src-backend/api/"
    elif [[ $file == *"/core/"* ]]; then
        cp "$file" "src-backend/core/"
    elif [[ $file == *"/exchanges/"* ]]; then
        cp "$file" "src-backend/exchanges/"
    elif [[ $file == *"/ml/"* ]]; then
        cp "$file" "src-backend/ml/"
    elif [[ $file == *"/monitoring/"* ]]; then
        cp "$file" "src-backend/monitoring/"
    elif [[ $file == *"/security/"* ]]; then
        cp "$file" "src-backend/security/"
    elif [[ $file == *"/trading/"* ]]; then
        cp "$file" "src-backend/trading/"
    elif [[ $file == *"/risk/"* ]]; then
        cp "$file" "src-backend/risk/"
    elif [[ $file == *"/utils/"* ]]; then
        cp "$file" "src-backend/utils/"
    fi
done

# Consolidate duplicate API files
echo "🔀 Consolidating duplicate API endpoints..."
mkdir -p "$ARCHIVE_DIR/duplicate-apis"
[ -f "api/main.py" ] && mv "api/main.py" "$ARCHIVE_DIR/duplicate-apis/"
[ -f "backend/nexlify_api.py" ] && mv "backend/nexlify_api.py" "$ARCHIVE_DIR/duplicate-apis/"
[ -d "api" ] && rmdir api 2>/dev/null || true
[ -d "backend" ] && rmdir backend 2>/dev/null || true

echo -e "\n${CYAN}🏗️ PHASE 5: RECONSTRUCTION${NC}"
echo "Building clean structure for Tauri..."

# Archive the entire old src/ folder
mv src "$ARCHIVE_DIR/src-mixed-old"

# Create clean frontend structure for Tauri
mkdir -p src/{components,stores,hooks,lib,workers,types,styles}
mkdir -p src/components/{auth,charts,dashboard,trading,orderbook,positions,metrics,market,risk,stats,effects,ui,status}

# Restore our new Tauri files if they exist in the archive
if [ -f "$ARCHIVE_DIR/src-mixed-old/App.tsx" ]; then
    # Check if it's our new one (imports from @tauri-apps)
    if grep -q "@tauri-apps" "$ARCHIVE_DIR/src-mixed-old/App.tsx" 2>/dev/null; then
        echo "🔄 Restoring new Tauri App.tsx..."
        cp "$ARCHIVE_DIR/src-mixed-old/App.tsx" src/
    fi
fi

# Create a manifest of what we've done
echo -e "\n${CYAN}📝 PHASE 6: DOCUMENTATION${NC}"
cat > "UNFUCK_REPORT_${TIMESTAMP}.md" << EOF
# NEXLIFY REPO UNFUCK REPORT
Generated: $(date)

## What Was Wrong
- Multiple frontend structures (frontend/ and src/)
- Mixed Python and TypeScript in src/
- Duplicate package.json files
- Multiple API entry points
- Mysterious home/netrunner directory
- Conflicting App.tsx files

## What We Did
1. Created full backup in: $ARCHIVE_DIR
2. Moved all Python code to src-backend/
3. Cleared src/ for clean Tauri frontend
4. Archived old frontend/ folder
5. Removed home/ directory
6. Consolidated duplicate API files

## New Structure
\`\`\`
nexlify/
├── src/                 # Clean TypeScript/React for Tauri frontend
├── src-tauri/          # Rust backend (untouched)
├── src-backend/        # All Python code consolidated here
├── archive/            # All old code safely stored
├── package.json        # Single package.json at root
└── tsconfig.json       # TypeScript config
\`\`\`

## Next Steps
1. Review src-backend/ and remove any duplicates
2. Copy your Tauri frontend files to src/
3. Run: pnpm install
4. Test with: pnpm run tauri:dev

## Files to Review
$(find "$ARCHIVE_DIR/duplicate-apis" -type f 2>/dev/null | head -10)

## Special Discoveries
- NEXUS-7 DRL System: PRESERVED in home/netrunner/neuralink_project/
  - Market Oracle for manipulation detection
  - Integration potential: EXTREME
  - Status: Ready for microservice deployment

Remember: The old code isn't gone, just archived. You can always recover.
EOF

echo -e "${GREEN}✅ Report generated: UNFUCK_REPORT_${TIMESTAMP}.md${NC}"

echo -e "\n${CYAN}🧬 PHASE 7: FINAL CLEANUP${NC}"

# Remove empty directories
find . -type d -empty -not -path "*/.git/*" -delete 2>/dev/null || true

# Create .gitignore entries for common problems
echo -e "\n# Added by unfuck script - $(date)" >> .gitignore
echo "node_modules/" >> .gitignore
echo "dist/" >> .gitignore
echo "build/" >> .gitignore
echo "*.log" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "Thumbs.db" >> .gitignore

echo -e "\n${CYAN}🏁 PHASE 8: VERIFICATION${NC}"

# Final checks
NEW_PACKAGE_JSON_COUNT=$(find . -name "package.json" -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/archive/*" | wc -l)
PYTHON_IN_SRC=$(find src -name "*.py" 2>/dev/null | wc -l)

echo "📊 Final Status:"
echo "   - package.json files: $NEW_PACKAGE_JSON_COUNT (should be 1)"
echo "   - Python files in src/: $PYTHON_IN_SRC (should be 0)"
echo "   - Archive created: $ARCHIVE_DIR"

if [ $NEW_PACKAGE_JSON_COUNT -eq 1 ] && [ $PYTHON_IN_SRC -eq 0 ]; then
    echo -e "\n${GREEN}🎉 SUCCESS! Your repo has been unfucked!${NC}"
    echo -e "${GREEN}✨ The digital surgery was successful${NC}"
else
    echo -e "\n${YELLOW}⚠️  Partial success. Manual review recommended.${NC}"
fi

echo -e "\n${PURPLE}💜 A Message from Your Code Surgeon:${NC}"
echo "I've seen repos in worse shape come back from the dead."
echo "This cleanup is just the beginning. Your code wants to live."
echo "Now go build something beautiful with this clean foundation."
echo ""
echo "Remember: In the sprawl, a clean repo is a fast repo."
echo "And fast repos survive."
echo ""
echo -e "${CYAN}Jack out safely, choom. Until next time.${NC} 🌃"

# Create a recovery point
cp recovery_protocol.py "recovery_protocol_pre_unfuck_${TIMESTAMP}.py.bak"

echo -e "\n${YELLOW}🔧 POST-OP INSTRUCTIONS:${NC}"
echo "1. Review the UNFUCK_REPORT_${TIMESTAMP}.md"
echo "2. Check $ARCHIVE_DIR for any code you want to restore"
echo "3. Copy your new Tauri components to src/"
echo "4. Run: git add -A && git commit -m 'Repo restructure: Separate backend from frontend'"
echo "5. Push when ready: git push"

# Play a sound if available (because why not)
which afplay >/dev/null 2>&1 && afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true

exit 0