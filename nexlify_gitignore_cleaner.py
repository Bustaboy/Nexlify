#!/usr/bin/env python3
"""
Nexlify Git Sanitizer - Keep Your Secrets in the Shadows
Cleans up your repo and updates .gitignore with everything that shouldn't be public
Because even netrunners need good OpSec
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import fnmatch

class GitSanitizer:
    def __init__(self, root_path="."):
        self.root = Path(root_path)
        self.gitignore_path = self.root / ".gitignore"
        self.existing_patterns = set()
        self.new_patterns = []
        self.files_to_remove = []
        
        # Patterns that should ALWAYS be in .gitignore
        self.essential_patterns = [
            # Build artifacts
            "target/",
            "dist/",
            "build/",
            "*.pyc",
            "__pycache__/",
            "*.pyo",
            "*.pyd",
            ".Python",
            "*.so",
            "*.dylib",
            "*.dll",
            "*.class",
            "*.log",
            
            # Dependencies
            "node_modules/",
            "venv/",
            "env/",
            ".env",
            ".venv",
            "pip-log.txt",
            "pip-delete-this-directory.txt",
            
            # IDE and editor files
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            "*~",
            ".project",
            ".classpath",
            ".settings/",
            "*.sublime-workspace",
            "*.sublime-project",
            
            # OS files
            ".DS_Store",
            ".DS_Store?",
            "._*",
            ".Spotlight-V100",
            ".Trashes",
            "ehthumbs.db",
            "Thumbs.db",
            "desktop.ini",
            
            # Sensitive data
            "*.key",
            "*.pem",
            "*.p12",
            "*.pfx",
            "*.cer",
            "*.crt",
            "*.der",
            "secrets/",
            "credentials/",
            ".env.local",
            ".env.*.local",
            
            # Test and coverage
            ".coverage",
            "*.cover",
            ".hypothesis/",
            ".pytest_cache/",
            "nosetests.xml",
            "coverage.xml",
            "*.lcov",
            ".nyc_output/",
            
            # Logs and databases
            "*.log",
            "*.sql",
            "*.sqlite",
            "*.sqlite3",
            "*.db",
            "logs/",
            
            # Temporary files
            "*.tmp",
            "*.temp",
            "*.bak",
            "*.backup",
            "*.old",
            "~*",
            
            # Rust/Tauri specific
            "Cargo.lock",  # For libraries
            "target/",
            "**/*.rs.bk",
            "*.pdb",
            
            # Project specific - based on your directory structure
            "backup_*/",
            "diagnostic_report_*.txt",
            "UNFUCK_REPORT_*.md",
            "src-tauri/target/",
            "src-tauri/gen/",
            "*.backup",
            "compose.yaml",  # If it contains secrets
            "docker-compose.override.yml",
            
            # Archive folders (old code)
            "archive/",  # Your old code archive
            
            # Python specific
            "*.egg-info/",
            "dist/",
            ".eggs/",
            "*.egg",
            
            # Cache directories
            ".cache/",
            ".parcel-cache/",
            ".next/",
            ".nuxt/",
            ".vuepress/dist/",
            ".serverless/",
            ".fusebox/",
            ".dynamodb/",
            
            # Package files that might contain sensitive data
            "pnpm-debug.log*",
            "yarn-debug.log*",
            "yarn-error.log*",
            "lerna-debug.log*",
            "npm-debug.log*",
        ]
        
    def load_existing_gitignore(self):
        """Load existing .gitignore patterns"""
        if self.gitignore_path.exists():
            with open(self.gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.existing_patterns.add(line)
            print(f"📖 Loaded {len(self.existing_patterns)} existing patterns from .gitignore")
        else:
            print("⚠️ No .gitignore found - will create one")
    
    def find_patterns_to_add(self):
        """Determine which patterns need to be added"""
        for pattern in self.essential_patterns:
            if pattern not in self.existing_patterns:
                self.new_patterns.append(pattern)
        
        print(f"\n🔍 Found {len(self.new_patterns)} patterns to add")
    
    def check_committed_files(self):
        """Check if any files matching our patterns are already in git"""
        print("\n🔍 Checking for files that shouldn't be in git...")
        
        try:
            # Get list of all tracked files
            result = subprocess.run(
                ["git", "ls-files"],
                capture_output=True,
                text=True,
                cwd=self.root
            )
            
            if result.returncode != 0:
                print("⚠️ Not a git repository or git not available")
                return
            
            tracked_files = result.stdout.strip().split('\n') if result.stdout else []
            
            # Check each tracked file against our patterns
            for file_path in tracked_files:
                for pattern in self.essential_patterns:
                    # Convert glob pattern for matching
                    if self._should_ignore(file_path, pattern):
                        self.files_to_remove.append(file_path)
                        break
            
            if self.files_to_remove:
                print(f"⚠️ Found {len(self.files_to_remove)} files that should be removed from git:")
                for f in self.files_to_remove[:10]:  # Show first 10
                    print(f"   - {f}")
                if len(self.files_to_remove) > 10:
                    print(f"   ... and {len(self.files_to_remove) - 10} more")
                    
        except Exception as e:
            print(f"⚠️ Error checking git files: {e}")
    
    def _should_ignore(self, file_path, pattern):
        """Check if a file matches a gitignore pattern"""
        # Handle directory patterns
        if pattern.endswith('/'):
            return file_path.startswith(pattern[:-1] + '/') or file_path + '/' == pattern
        
        # Handle glob patterns
        if '*' in pattern or '?' in pattern or '[' in pattern:
            return fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern)
        
        # Handle exact matches
        return file_path == pattern or os.path.basename(file_path) == pattern
    
    def update_gitignore(self):
        """Update .gitignore with new patterns"""
        if not self.new_patterns:
            print("\n✅ .gitignore is already comprehensive")
            return
        
        print(f"\n📝 Adding {len(self.new_patterns)} patterns to .gitignore")
        
        # Read existing content
        existing_content = ""
        if self.gitignore_path.exists():
            with open(self.gitignore_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # Prepare new content
        with open(self.gitignore_path, 'w', encoding='utf-8') as f:
            # Write existing content
            if existing_content and not existing_content.endswith('\n'):
                existing_content += '\n'
            f.write(existing_content)
            
            # Add our new patterns with categories
            if existing_content:
                f.write("\n")
            
            f.write("# ===== Nexlify Git Sanitizer Additions =====\n")
            f.write(f"# Added by sanitizer on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group patterns by category
            categories = {
                "Build Artifacts": ["target/", "dist/", "build/", "*.pyc", "__pycache__/"],
                "Dependencies": ["node_modules/", "venv/", ".env", ".venv"],
                "IDE Files": [".vscode/", ".idea/", "*.swp"],
                "OS Files": [".DS_Store", "Thumbs.db", "desktop.ini"],
                "Sensitive Data": ["*.key", "*.pem", "secrets/", "credentials/"],
                "Temporary Files": ["*.tmp", "*.bak", "backup_*/", "diagnostic_report_*.txt"],
                "Archives": ["archive/"],
            }
            
            written_patterns = set()
            
            for category, patterns in categories.items():
                category_patterns = [p for p in self.new_patterns if p in patterns and p not in written_patterns]
                if category_patterns:
                    f.write(f"# {category}\n")
                    for pattern in category_patterns:
                        f.write(f"{pattern}\n")
                        written_patterns.add(pattern)
                    f.write("\n")
            
            # Write any remaining patterns
            remaining = [p for p in self.new_patterns if p not in written_patterns]
            if remaining:
                f.write("# Other patterns\n")
                for pattern in remaining:
                    f.write(f"{pattern}\n")
        
        print("✅ .gitignore updated successfully")
    
    def create_removal_script(self):
        """Create a script to remove files from git"""
        if not self.files_to_remove:
            return
        
        script_name = "remove_from_git.sh"
        
        with open(script_name, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n")
            f.write("# Script to remove files that shouldn't be in git\n")
            f.write("# Generated by Nexlify Git Sanitizer\n\n")
            
            f.write("echo '🧹 Removing files from git (keeping local copies)...'\n\n")
            
            for file_path in self.files_to_remove:
                f.write(f"git rm --cached '{file_path}' 2>/dev/null\n")
            
            f.write("\necho '✅ Files removed from git tracking'\n")
            f.write("echo '📝 Now commit these changes:'\n")
            f.write("echo '   git commit -m \"chore: remove files that should be gitignored\"'\n")
        
        # Make script executable on Unix-like systems
        try:
            os.chmod(script_name, 0o755)
        except:
            pass
        
        print(f"\n💾 Created removal script: {script_name}")
        print("   Run this script to remove files from git while keeping local copies")
        print("   Windows: Use Git Bash to run ./remove_from_git.sh")
    
    def run(self):
        """Execute the sanitization process"""
        print("🧹 NEXLIFY GIT SANITIZER")
        print("=" * 50)
        print("Scanning for files that don't belong in the repo...\n")
        
        # Load existing .gitignore
        self.load_existing_gitignore()
        
        # Find patterns to add
        self.find_patterns_to_add()
        
        # Check for already committed files
        self.check_committed_files()
        
        # Update .gitignore
        self.update_gitignore()
        
        # Create removal script if needed
        if self.files_to_remove:
            self.create_removal_script()
        
        # Final summary
        print("\n" + "=" * 50)
        print("📊 SANITIZATION COMPLETE")
        print("=" * 50)
        
        if self.new_patterns:
            print(f"✅ Added {len(self.new_patterns)} patterns to .gitignore")
        
        if self.files_to_remove:
            print(f"⚠️ Found {len(self.files_to_remove)} files to remove from git")
            print("   Run ./remove_from_git.sh to clean them")
        
        print("\n🎯 Next steps:")
        print("1. Review the updated .gitignore")
        print("2. Run ./remove_from_git.sh if files need removal")
        print("3. Commit your changes")
        print("\nStay clean in the Net, choom! 🌃")

if __name__ == "__main__":
    sanitizer = GitSanitizer()
    sanitizer.run()
