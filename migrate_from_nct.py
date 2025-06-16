#!/usr/bin/env python3
"""
Migrate modules from Night-City-Trader to Nexlify
Handles rebranding and path updates
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import argparse


class NCTMigrator:
    """Migrates Night-City-Trader modules to Nexlify"""
    
    def __init__(self, nct_path: str, nexlify_path: str = "."):
        self.nct_path = Path(nct_path)
        self.nexlify_path = Path(nexlify_path)
        self.logger = self._setup_logger()
        
        # Module mapping
        self.module_map = {
            # Core modules to migrate
            "arasaka_neural_net.py": "src/core/nexlify_neural_net.py",
            "error_handler.py": "src/core/error_handler.py",
            "smart_launcher.py": "src/smart_launcher.py",
            "cyber_gui.py": "src/gui/nexlify_enhanced_gui.py",
            "utils_module.py": "src/utils/utils_module.py",
            
            # Optional modules
            "nexlify_advanced_security.py": "src/security/nexlify_advanced_security.py",
            "nexlify_audit_trail.py": "src/audit/nexlify_audit_trail.py",
            "nexlify_mobile_api.py": "src/mobile/nexlify_mobile_api.py",
            "nexlify_ai_companion.py": "src/ai/nexlify_ai_companion.py",
            "nexlify_dex_integration.py": "src/trading/nexlify_dex_integration.py",
            "nexlify_multi_strategy.py": "src/strategies/nexlify_multi_strategy.py",
            "nexlify_predictive_features.py": "src/ml/nexlify_predictive_features.py",
            "nexlify_advanced_backtesting.py": "src/trading/nexlify_advanced_backtesting.py",
            "nexlify_cyberpunk_effects.py": "src/gui/nexlify_cyberpunk_effects.py",
        }
        
        # Text replacements for rebranding
        self.replacements = [
            # Branding
            (r"Night[\s-]?City[\s-]?Trader", "Nexlify"),
            (r"night[\s-]?city[\s-]?trader", "nexlify"),
            (r"NCT", "NX"),
            (r"nct", "nx"),
            
            # Class names
            (r"ArasakaNeuralNet", "NexlifyNeuralNet"),
            (r"NightCityErrorHandler", "NexlifyErrorHandler"),
            (r"CyberGUI", "NexlifyEnhancedGUI"),
            
            # Config files
            (r"neural_config\.json", "enhanced_config.json"),
            (r"night_city_config", "nexlify_config"),
            
            # Imports (careful with paths)
            (r"from arasaka_neural_net", "from src.core.nexlify_neural_net"),
            (r"from cyber_gui", "from src.gui.nexlify_enhanced_gui"),
            (r"from error_handler", "from src.core.error_handler"),
            (r"from utils_module", "from src.utils.utils_module"),
            
            # Comments and strings
            (r"Arasaka", "Nexlify"),
            (r"arasaka", "nexlify"),
        ]
    
    def _setup_logger(self) -> logging.Logger:
        """Setup colored logger"""
        logger = logging.getLogger("NCTMigrator")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def validate_paths(self) -> bool:
        """Validate source and destination paths"""
        if not self.nct_path.exists():
            self.logger.error(f"Night-City-Trader path not found: {self.nct_path}")
            return False
        
        if not self.nexlify_path.exists():
            self.logger.error(f"Nexlify path not found: {self.nexlify_path}")
            return False
        
        return True
    
    def migrate_module(self, source_file: str, dest_file: str) -> bool:
        """Migrate a single module with rebranding"""
        source_path = self.nct_path / source_file
        dest_path = self.nexlify_path / dest_file
        
        if not source_path.exists():
            self.logger.warning(f"Source not found: {source_file}")
            return False
        
        try:
            # Read source content
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply replacements
            original_content = content
            for pattern, replacement in self.replacements:
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            
            # Add migration header
            header = f'''"""
Migrated from Night-City-Trader to Nexlify
Original: {source_file}
Note: This is a rebranded version - verify imports and configuration
"""

'''
            
            # Don't add header if already present
            if "Migrated from Night-City-Trader" not in content:
                content = header + content
            
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if destination exists
            if dest_path.exists():
                response = input(f"{dest_file} exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    self.logger.info(f"Skipped: {dest_file}")
                    return False
            
            # Write migrated content
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Report changes
            if content != original_content:
                self.logger.info(f"✅ Migrated and rebranded: {source_file} -> {dest_file}")
            else:
                self.logger.info(f"✅ Migrated: {source_file} -> {dest_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to migrate {source_file}: {e}")
            return False
    
    def migrate_configs(self) -> bool:
        """Migrate configuration files"""
        self.logger.info("📋 Migrating configurations...")
        
        config_map = {
            "neural_config.json": "config/neural_config_backup.json",
            "requirements.txt": "requirements_nct.txt",
        }
        
        for source, dest in config_map.items():
            source_path = self.nct_path / source
            dest_path = self.nexlify_path / dest
            
            if source_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                self.logger.info(f"✅ Backed up: {source} -> {dest}")
        
        return True
    
    def create_import_adapter(self) -> bool:
        """Create adapter for old imports"""
        self.logger.info("🔌 Creating import adapter...")
        
        adapter_content = '''"""
Import adapter for Night-City-Trader compatibility
Allows gradual migration of imports
"""

import sys
from pathlib import Path

# Add compatibility mappings
IMPORT_MAP = {
    "arasaka_neural_net": "src.core.nexlify_neural_net",
    "cyber_gui": "src.gui.nexlify_enhanced_gui",
    "error_handler": "src.core.error_handler",
    "utils_module": "src.utils.utils_module",
}

class NCTCompatibilityFinder:
    """Import hook for NCT compatibility"""
    
    def find_module(self, fullname, path=None):
        if fullname in IMPORT_MAP:
            return self
        return None
    
    def load_module(self, fullname):
        if fullname in IMPORT_MAP:
            # Redirect to new module
            new_name = IMPORT_MAP[fullname]
            __import__(new_name)
            module = sys.modules[new_name]
            sys.modules[fullname] = module
            return module
        raise ImportError(f"No module named {fullname}")

# Install the import hook
sys.meta_path.insert(0, NCTCompatibilityFinder())

print("NCT compatibility layer activated")
'''
        
        adapter_path = self.nexlify_path / "src" / "nct_compat.py"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(adapter_path, 'w') as f:
            f.write(adapter_content)
        
        self.logger.info("✅ Created import adapter")
        return True
    
    def create_migration_report(self) -> None:
        """Create a migration report"""
        report_path = self.nexlify_path / "MIGRATION_REPORT.md"
        
        report = f"""# Night-City-Trader to Nexlify Migration Report

## Migration Date
{Path.ctime(Path(__file__))}

## Migrated Modules
"""
        
        for source, dest in self.module_map.items():
            dest_path = self.nexlify_path / dest
            if dest_path.exists():
                report += f"- ✅ {source} -> {dest}\n"
            else:
                report += f"- ❌ {source} (not found)\n"
        
        report += """
## Rebranding Applied
- Night City Trader -> Nexlify
- ArasakaNeuralNet -> NexlifyNeuralNet
- neural_config.json -> enhanced_config.json
- All imports updated

## Next Steps
1. Review migrated modules for import errors
2. Update any hardcoded paths
3. Test each module individually
4. Update configuration files
5. Run full system test

## Notes
- Original configs backed up with _backup suffix
- Import adapter created for compatibility
- Some manual adjustments may be needed
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"📄 Migration report saved: {report_path}")
    
    def run(self, modules: List[str] = None) -> bool:
        """Run the migration process"""
        if not self.validate_paths():
            return False
        
        print(f"\n🚀 Migrating Night-City-Trader to Nexlify")
        print(f"Source: {self.nct_path}")
        print(f"Destination: {self.nexlify_path}\n")
        
        # Select modules to migrate
        if modules:
            module_list = [(m, self.module_map.get(m, f"src/{m}")) 
                          for m in modules if m in self.module_map]
        else:
            module_list = list(self.module_map.items())
        
        # Migrate modules
        success_count = 0
        for source, dest in module_list:
            if self.migrate_module(source, dest):
                success_count += 1
        
        # Migrate configs
        self.migrate_configs()
        
        # Create compatibility layer
        self.create_import_adapter()
        
        # Create report
        self.create_migration_report()
        
        print(f"\n✅ Migration complete: {success_count}/{len(module_list)} modules")
        print("📖 See MIGRATION_REPORT.md for details")
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Migrate Night-City-Trader modules to Nexlify"
    )
    parser.add_argument(
        "nct_path",
        help="Path to Night-City-Trader source"
    )
    parser.add_argument(
        "--nexlify-path",
        default=".",
        help="Path to Nexlify destination (default: current directory)"
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        help="Specific modules to migrate (default: all)"
    )
    
    args = parser.parse_args()
    
    migrator = NCTMigrator(args.nct_path, args.nexlify_path)
    success = migrator.run(args.modules)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
