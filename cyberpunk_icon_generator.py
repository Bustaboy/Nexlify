#!/usr/bin/env python3
"""
Nexlify Icon Generator - Quick Chrome Fix
Generates placeholder icons so you can compile NOW
"""

import os
from pathlib import Path

def create_placeholder_icons():
    """Generate minimal icons to get the build working"""
    
    # Create icons directory
    icons_dir = Path("src-tauri/icons")
    icons_dir.mkdir(exist_ok=True)
    
    # Simple 1x1 pixel ICO file (smallest valid ICO)
    ico_header = b'\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00\x20\x00'
    ico_header += b'\x30\x00\x00\x00\x16\x00\x00\x00'
    ico_dib = b'\x28\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x01\x00\x20\x00'
    ico_dib += b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    ico_dib += b'\x00\x00\x00\x00\x00\x00\x00\x00'
    ico_pixel = b'\xFF\x00\xFF\xFF\x00\x00\x00\x00'  # Magenta pixel + mask
    
    ico_data = ico_header + ico_dib + ico_pixel
    
    # Write ICO file
    ico_path = icons_dir / "icon.ico"
    with open(ico_path, 'wb') as f:
        f.write(ico_data)
    
    print(f"✓ Created {ico_path}")
    
    # Create placeholder PNG files (1x1 transparent pixel)
    png_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00'
        b'\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\r'
        b'IDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x00\x00\x00\x00'
        b'IEND\xaeB`\x82'
    )
    
    # Create required PNG sizes
    png_files = [
        "32x32.png",
        "128x128.png", 
        "128x128@2x.png",
        "icon.png"
    ]
    
    for png_file in png_files:
        png_path = icons_dir / png_file
        with open(png_path, 'wb') as f:
            f.write(png_data)
        print(f"✓ Created {png_path}")
    
    # Create ICNS placeholder (Mac) - just use PNG data
    icns_path = icons_dir / "icon.icns"
    with open(icns_path, 'wb') as f:
        f.write(png_data)  # Not a real ICNS but will work for now
    print(f"✓ Created {icns_path}")
    
    print("\n🎨 Placeholder icons created!")
    print("   These are just to get you compiling.")
    print("   Replace with real Nexlify icons later.")

if __name__ == "__main__":
    create_placeholder_icons()
