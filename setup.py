#!/usr/bin/env python3
"""
Nexlify - AI-Powered Cryptocurrency Trading Platform
Setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
llong_description = (this_directory / "README.md").read_text(encoding='utf-8') 

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').splitlines()

# Filter out comments and empty lines
requirements = [
    req.strip() for req in requirements
    if req.strip() and not req.startswith('#')
]

setup(
    name="nexlify",
    version="2.0.7.7",
    author="Nexlify Development Team",
    description="AI-Powered Cryptocurrency Trading Platform with advanced risk management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bustaboy/Nexlify",
    packages=find_packages(exclude=["tests", "tests.*", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.1",
            "black>=23.9.1",
            "pylint>=2.17.7",
            "mypy>=1.5.1",
            "isort>=5.12.0",
        ],
        "sound": [
            "playsound==1.3.0",
            "plyer==2.1.0",
        ],
        "windows": [
            "win10toast==0.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "nexlify=scripts.nexlify_launcher:main",
            "nexlify-setup=scripts.setup_nexlify:main",
        ],
    },
    include_package_data=True,
    package_data={
        "nexlify": ["config/*.json"],
    },
    zip_safe=False,
)
