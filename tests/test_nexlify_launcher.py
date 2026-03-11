"""Tests for root-level Nexlify launcher preflight behavior."""

from pathlib import Path

import nexlify_launcher


def test_required_files_exist_in_repo_layout():
    missing = nexlify_launcher.missing_files()
    assert missing == []


def test_missing_modules_helper_detects_nonexistent_module():
    missing = nexlify_launcher.missing_modules(["definitely_not_a_real_module_123"])
    assert missing == ["definitely_not_a_real_module_123"]


def test_launcher_root_points_to_repo_root():
    assert (nexlify_launcher.PROJECT_ROOT / "README.md").exists()
    assert isinstance(nexlify_launcher.PROJECT_ROOT, Path)
