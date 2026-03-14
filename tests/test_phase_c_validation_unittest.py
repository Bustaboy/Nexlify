import unittest
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch
import subprocess

from scripts.phase_c_validation import (
    CheckResult,
    compute_exit_code,
    determine_overall_status,
    get_recommendations,
    parse_args,
    run_command,
)


class PhaseCValidationLogicTests(unittest.TestCase):
    def test_determine_overall_status_precedence(self):
        self.assertEqual(determine_overall_status({"PASS": 0, "WARN": 0, "FAIL": 1, "SKIP": 0}), "FAIL")
        self.assertEqual(determine_overall_status({"PASS": 1, "WARN": 2, "FAIL": 0, "SKIP": 0}), "WARN")
        self.assertEqual(determine_overall_status({"PASS": 1, "WARN": 0, "FAIL": 0, "SKIP": 3}), "SKIP")
        self.assertEqual(determine_overall_status({"PASS": 4, "WARN": 0, "FAIL": 0, "SKIP": 0}), "PASS")
        self.assertEqual(determine_overall_status({"PASS": 2}), "PASS")

    def test_compute_exit_code_default_and_strict(self):
        args_default = SimpleNamespace(strict=False, fail_on_skip=False)
        args_strict = SimpleNamespace(strict=True, fail_on_skip=False)
        args_strict_skip = SimpleNamespace(strict=True, fail_on_skip=True)

        fail = [CheckResult("x", "X", "FAIL", "", 0.0)]
        warn = [CheckResult("w", "W", "WARN", "", 0.0)]
        skip = [CheckResult("s", "S", "SKIP", "", 0.0)]

        self.assertEqual(compute_exit_code(args_default, [], [], []), (0, "default"))
        self.assertEqual(compute_exit_code(args_default, fail, [], []), (1, "default"))
        self.assertEqual(compute_exit_code(args_strict, [], warn, []), (1, "strict"))
        self.assertEqual(compute_exit_code(args_strict_skip, [], [], skip), (1, "strict + fail-on-skip"))

    def test_recommendations_include_expected_hints(self):
        results = [
            CheckResult("python_version", "Python version", "WARN", "needs 3.12", 0.0),
            CheckResult("import_probe", "Import probe", "WARN", "Missing: torch", 0.0),
            CheckResult("preflight", "Preflight", "SKIP", "skip", 0.0),
        ]
        recs = get_recommendations({"PASS": 0, "WARN": 2, "FAIL": 0, "SKIP": 1}, results)
        text = "\n".join(recs)
        self.assertIn("Python 3.12+", text)
        self.assertIn("pip install -r requirements.txt", text)
        self.assertIn("--force-heavy", text)

    def test_parse_args_rejects_non_positive_max_detail_lines(self):
        with patch("sys.argv", ["phase_c_validation.py", "--max-detail-lines", "0"]):
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_recommendations_handle_sparse_counts(self):
        results = [CheckResult("python_version", "Python version", "PASS", "ok", 0.0)]
        recs = get_recommendations({"PASS": 1}, results)
        self.assertTrue(any("Environment looks healthy" in item for item in recs))

    def test_run_command_handles_oserror(self):
        with patch("subprocess.run", side_effect=OSError("boom")):
            result = run_command("broken", ["fake-cmd"], 10, 30)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("Unable to run command", result.details)

    def test_run_command_handles_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["slow"], timeout=1, output="", stderr="took too long")):
            result = run_command("slow check", ["slow"], 10, 1)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("timed out", result.details.lower())


if __name__ == "__main__":
    unittest.main()
