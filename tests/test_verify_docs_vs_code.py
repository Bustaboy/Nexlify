"""Unit tests for docs/code drift verification helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_verify_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "verify_docs_vs_code.py"
    spec = spec_from_file_location("verify_docs_vs_code", module_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_paths_helper_detects_missing_path(tmp_path: Path):
    verify_docs_vs_code = _load_verify_module()
    missing_target = tmp_path / "not_there.txt"
    missing = verify_docs_vs_code.missing_paths([missing_target])
    assert missing == [str(missing_target)]


def test_missing_module_specs_detects_nonexistent_module():
    verify_docs_vs_code = _load_verify_module()
    missing = verify_docs_vs_code.missing_module_specs(["definitely_not_real_pkg_xyz"])
    assert missing == ["definitely_not_real_pkg_xyz"]


def test_missing_module_specs_finds_existing_local_module_spec():
    verify_docs_vs_code = _load_verify_module()
    missing = verify_docs_vs_code.missing_module_specs(["nexlify.financial.nexlify_tax_reporter"])
    assert missing == []
