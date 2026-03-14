#!/usr/bin/env python3
"""Phase C validation runner for Python 3.12 migration.

Usage:
  python scripts/phase_c_validation.py
  python scripts/phase_c_validation.py --strict
  python scripts/phase_c_validation.py --json-out phase_c_report.json
  python scripts/phase_c_validation.py --md-out phase_c_report.md
  python scripts/phase_c_validation.py --force-heavy
  python scripts/phase_c_validation.py --max-detail-lines 40
  python scripts/phase_c_validation.py --summary-only
  python scripts/phase_c_validation.py --fail-on-skip
  python scripts/phase_c_validation.py --command-timeout 120
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
from typing import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

MODULES = ["tensorflow", "torch", "torchvision", "ccxt", "pandas", "numpy", "PyQt5", "web3"]


@dataclass
class CheckResult:
    check_id: str
    name: str
    status: str  # PASS / WARN / FAIL / SKIP
    details: str
    duration_s: float = 0.0


def _truncate_details(details: str, max_lines: int) -> str:
    max_lines = max(1, max_lines)
    lines = details.splitlines()
    if len(lines) <= max_lines:
        return details
    keep = lines[:max_lines]
    keep.append(f"... (truncated {len(lines) - max_lines} lines)")
    return "\n".join(keep)


def _classify_env_limited_failure(details: str) -> str:
    lowered = details.lower()
    env_limited_markers = [
        "no module named",
        "missing dependency",
        "needs 3.12+",
    ]
    if any(marker in lowered for marker in env_limited_markers):
        return "WARN"
    return "FAIL"


def _to_check_id(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def run_command(name: str, cmd: Sequence[str], max_detail_lines: int, timeout_s: int) -> CheckResult:
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        details = _truncate_details(
            f"Command timed out after {timeout_s}s: {' '.join(cmd)}\n{(exc.stderr or exc.stdout or '').strip()}",
            max_detail_lines,
        )
        return CheckResult(_to_check_id(name), name, "FAIL", details, duration)
    except OSError as exc:
        duration = time.perf_counter() - started
        details = _truncate_details(f"Unable to run command: {exc}", max_detail_lines)
        return CheckResult(_to_check_id(name), name, "FAIL", details, duration)
    duration = time.perf_counter() - started
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    details = out if out else err
    details = _truncate_details(details or f"Exit code {proc.returncode}", max_detail_lines)
    if proc.returncode == 0:
        return CheckResult(_to_check_id(name), name, "PASS", details or "OK", duration)
    status = _classify_env_limited_failure(details)
    return CheckResult(_to_check_id(name), name, status, details, duration)


def check_python_version() -> CheckResult:
    version = platform.python_version()
    major, minor, *_ = sys.version_info
    if (major, minor) >= (3, 12):
        return CheckResult("python_version", "Python version", "PASS", f"Python {version}")
    return CheckResult("python_version", "Python version", "WARN", f"Python {version} (needs 3.12+)")


def check_imports() -> tuple[CheckResult, list[str]]:
    missing = [m for m in MODULES if importlib.util.find_spec(m) is None]
    if missing:
        return CheckResult("import_probe", "Import probe", "WARN", f"Missing: {', '.join(missing)}"), missing
    return CheckResult("import_probe", "Import probe", "PASS", "All required modules discoverable"), []


def parse_args() -> argparse.Namespace:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be >= 1")
        return parsed

    parser = argparse.ArgumentParser(description="Run Phase C validation checks.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero on WARN/FAIL (default only fails on FAIL).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write JSON report.",
    )
    parser.add_argument(
        "--md-out",
        default="",
        help="Optional path to write Markdown summary report.",
    )
    parser.add_argument(
        "--force-heavy",
        action="store_true",
        help="Run preflight/GPU checks even if core imports are missing.",
    )
    parser.add_argument(
        "--max-detail-lines",
        type=positive_int,
        default=80,
        help="Maximum lines to print/store per command output before truncating.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print compact summary (status + timing) without full per-check details.",
    )
    parser.add_argument(
        "--command-timeout",
        type=positive_int,
        default=180,
        help="Timeout in seconds for each subprocess check.",
    )
    parser.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="Treat SKIP as a failure condition for exit-code purposes.",
    )
    return parser.parse_args()


def _status_emoji(status: str) -> str:
    return {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(status, "•")


def determine_overall_status(counts: dict[str, int]) -> str:
    if counts.get("FAIL", 0) > 0:
        return "FAIL"
    if counts.get("WARN", 0) > 0:
        return "WARN"
    if counts.get("SKIP", 0) > 0:
        return "SKIP"
    return "PASS"


def get_recommendations(counts: dict[str, int], results: list[CheckResult]) -> list[str]:
    recommendations: list[str] = []

    py_warn = any(r.check_id == "python_version" and r.status == "WARN" for r in results)
    import_warn = any(r.check_id == "import_probe" and r.status == "WARN" for r in results)
    preflight_skip = any(r.check_id == "preflight" and r.status == "SKIP" for r in results)

    if py_warn:
        recommendations.append("Use Python 3.12+ before treating Phase C as complete.")
    if import_warn:
        recommendations.append("Install full dependencies: pip install -r requirements.txt")
    if preflight_skip:
        recommendations.append("After dependencies are installed, rerun with --force-heavy to execute preflight/GPU checks.")
    if counts.get("FAIL", 0) > 0:
        recommendations.append("Investigate FAIL entries first; strict mode should remain failing until resolved.")
    elif counts.get("WARN", 0) == 0 and counts.get("SKIP", 0) == 0:
        recommendations.append("Environment looks healthy for Phase C; proceed to training smoke test.")

    return recommendations


def print_recommendations(recommendations: list[str]) -> None:
    print("\nRecommended next steps:")
    for item in recommendations:
        print(f"- {item}")


def write_markdown_report(
    path: str,
    counts: dict[str, int],
    overall_status: str,
    recommendations: list[str],
    results: list[CheckResult],
    args: argparse.Namespace,
    exit_code: int,
    exit_policy: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Phase C Validation Report",
        "",
        f"- Timestamp (UTC): `{now}`",
        f"- Python: `{platform.python_version()}`",
        f"- Strict mode: `{args.strict}`",
        f"- Force heavy: `{args.force_heavy}`",
        f"- Fail on skip: `{args.fail_on_skip}`",
        f"- Command timeout (s): `{args.command_timeout}`",
        f"- Exit policy: `{exit_policy}`",
        f"- Exit code: `{exit_code}`",
        "",
        "## Summary",
        "",
        f"- Overall status: `{overall_status}`",
        "",
        "## Summary Counts",
        "",
        f"- ✅ PASS: {counts['PASS']}",
        f"- ⚠️ WARN: {counts['WARN']}",
        f"- ❌ FAIL: {counts['FAIL']}",
        f"- ⏭️ SKIP: {counts['SKIP']}",
        "",
        "## Recommended Next Steps",
        "",
    ]
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.extend(["", "## Checks", ""])

    for r in results:
        lines.append(f"### {_status_emoji(r.status)} {r.name}")
        lines.append(f"- Check ID: `{r.check_id}`")
        lines.append(f"- Status: `{r.status}`")
        lines.append(f"- Duration: `{r.duration_s:.2f}s`")
        lines.append("")
        lines.append("```text")
        lines.append(r.details)
        lines.append("```")
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")




def compute_exit_code(args: argparse.Namespace, failures: list[CheckResult], warnings: list[CheckResult], skips: list[CheckResult]) -> tuple[int, str]:
    if args.strict:
        if args.fail_on_skip:
            return (1 if (failures or warnings or skips) else 0, "strict + fail-on-skip")
        return (1 if (failures or warnings) else 0, "strict")

    if args.fail_on_skip:
        return (1 if (failures or skips) else 0, "default + fail-on-skip")
    return (1 if failures else 0, "default")


def main() -> int:
    args = parse_args()
    results: list[CheckResult] = []

    results.append(check_python_version())
    results.append(run_command("pip check", [sys.executable, "-m", "pip", "check"], args.max_detail_lines, args.command_timeout))
    import_result, missing = check_imports()
    results.append(import_result)

    skip_heavy = bool(missing) and not args.force_heavy
    if skip_heavy:
        reason = "Skipped because required imports are missing; rerun with --force-heavy once dependencies are installed"
        results.append(CheckResult("preflight", "Preflight", "SKIP", reason, 0.0))
        results.append(CheckResult("gpu_verify_quick", "GPU verify (quick)", "SKIP", reason, 0.0))
    else:
        results.append(
            run_command(
                "Preflight",
                [sys.executable, "nexlify_preflight_checker.py", "--symbol", "BTC/USDT", "--automated"],
                args.max_detail_lines,
                args.command_timeout,
            )
        )
        results.append(
            run_command(
                "GPU verify (quick)",
                [sys.executable, "scripts/verify_gpu_training.py", "--quick"],
                args.max_detail_lines,
                args.command_timeout,
            )
        )

    print("\n=== Phase C Validation Summary ===")
    for r in results:
        if args.summary_only:
            print(f"[{r.status}] {r.name} ({r.duration_s:.2f}s)")
        else:
            print(f"[{r.status}] {r.name} ({r.duration_s:.2f}s): {r.details}")

    counts = {k: sum(1 for r in results if r.status == k) for k in ["PASS", "WARN", "FAIL", "SKIP"]}
    overall_status = determine_overall_status(counts)
    recommendations = get_recommendations(counts, results)

    print(f"\nCounts: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']} SKIP={counts['SKIP']}")
    print(f"Overall status: {overall_status}")
    print_recommendations(recommendations)

    failures = [r for r in results if r.status == "FAIL"]
    warnings = [r for r in results if r.status == "WARN"]
    skips = [r for r in results if r.status == "SKIP"]
    exit_code, exit_policy = compute_exit_code(args, failures, warnings, skips)
    print(f"Exit policy: {exit_policy} -> exit_code={exit_code}")

    if args.json_out:
        payload = {
            "summary": counts,
            "overall_status": overall_status,
            "strict_mode": args.strict,
            "force_heavy": args.force_heavy,
            "fail_on_skip": args.fail_on_skip,
            "command_timeout": args.command_timeout,
            "exit_policy": exit_policy,
            "exit_code": exit_code,
            "python": platform.python_version(),
            "recommended_next_steps": recommendations,
            "results": [asdict(r) for r in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote JSON report: {args.json_out}")

    if args.md_out:
        write_markdown_report(args.md_out, counts, overall_status, recommendations, results, args, exit_code, exit_policy)
        print(f"Wrote Markdown report: {args.md_out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
