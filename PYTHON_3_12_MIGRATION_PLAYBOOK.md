# Python 3.12 Migration Playbook (Living Document)

This is a **living document** for migrating Nexlify to a **Python 3.12-only** baseline and optimizing for NVIDIA-backed training.

---

## 1) Goal

Move the project to:
- Python **3.12 only**
- A consistent dependency + documentation baseline
- Validated compatibility for ML/RL + GPU workflows

---

## 2) Current Baseline Snapshot

Known inconsistencies to resolve:
- Packaging baseline has been moved to Python 3.12+ (continue aligning remaining docs/scripts).
- Core scripts now enforce 3.12+; continue scanning for stragglers.
- Continue periodic scans for stale version mentions in long-tail docs.
- Some PyTorch notes are stale relative to current requirements.

---

## 3) Execution Plan

## 3A) Current Phase Status (Quick View)

- ✅ **Phase A — Version Policy & Metadata:** Complete
- ✅ **Phase B — Script/Runtime Guards:** Complete for core runtime paths
- 🟨 **Phase C — Dependency & Compatibility Validation:** In progress (partial)
- 🟨 **Phase D — NVIDIA Optimization Pass:** In progress (container-blocked)
- ✅ **Phase E — Documentation Cleanup:** Complete (active docs)
- 🟨 **Phase F — Release Readiness:** In progress (handoff prepared)

> Legend: ✅ complete · 🟨 in progress/partial · ⬜ not started
> Last status refresh: 2026-03-14 (phase F prep #1)

### Phase A — Version Policy & Metadata
- [x] Set `python_requires` to `>=3.12`
- [x] Remove Python 3.11 classifier from package metadata
- [x] Keep Python 3.12 classifier
- [x] Add/update one canonical “supported Python version” section in docs

### Phase B — Script/Runtime Guards
- [x] Update all script version checks from 3.11+ to 3.12+
- [x] Verify launcher/setup shell/batch scripts show consistent messaging
- [x] Ensure preflight/setup flows fail fast with clear 3.12 guidance

### Phase C — Dependency & Compatibility Validation
- [ ] Create clean Python 3.12 venv *(blocked in current container: Python 3.10 runtime)*
- [ ] Install `requirements.txt` *(blocked in current container: deps not installed)*
- [x] Run `pip check`
- [ ] Run import smoke tests for key packages *(attempted; all missing in current environment)*:
  - `tensorflow`, `torch`, `torchvision`, `ccxt`, `pandas`, `numpy`, `PyQt5`, `web3`
- [ ] Run project preflight checks *(attempted; blocked due to missing deps in environment; see `scripts/phase_c_validation.py`)*
- [ ] Run at least one short training smoke test

### Phase D — NVIDIA Optimization Pass
- [ ] Validate CUDA/NVIDIA driver compatibility for selected torch build
- [x] Decide default profile:
  - local GPU training profile *(Primary)*
  - CI or CPU-only profile *(Fallback)*
- [ ] Evaluate optional accelerators:
  - `onnxruntime-gpu`
  - `torch-tensorrt`
- [ ] Update docs with tested install commands and expected verification output

### Phase E — Documentation Cleanup
- [x] Align all “Python required” statements to 3.12+ (core docs/scripts)
- [x] Remove stale version references for torch/tensorflow/CUDA instructions
- [x] Add troubleshooting matrix (version mismatch, CUDA mismatch, wheel mismatch)

### Phase F — Release Readiness
- [ ] Capture known-good environment tuple (Python, CUDA, driver, torch)
- [ ] Freeze tested dependency set (constraints/lock strategy selected and recorded)
- [ ] Record final migration status and open follow-up tasks

---

## 4) Agent Prompts (Reusable)

Use these prompts for iterative execution and verification.

### Prompt A — Perform migration edits

> Migrate this repo to Python 3.12-only.  
> Tasks:  
> 1) Update setup metadata (`python_requires`, classifiers) to 3.12-only.  
> 2) Update all version gates in scripts/docs from 3.11+ (or older) to 3.12+.  
> 3) Find and fix conflicting version statements across docs.  
> 4) Keep edits minimal and consistent.  
> 5) Run checks and show exact command outputs.  
> 6) Commit changes and open a PR with a concise migration summary.

### Prompt B — Compatibility audit (Python 3.12)

> Run a full Python 3.12 compatibility audit for this repo.  
> Steps:  
> - Create a clean 3.12 virtualenv.  
> - Install `requirements.txt`.  
> - Run `pip check`.  
> - Run import smoke tests for: tensorflow, torch, torchvision, PyQt5, web3, ccxt.  
> - Run project preflight checks.  
> - Return a table: package, version, status, issue, fix.

### Prompt C — NVIDIA-focused optimization suggestions

> Review dependencies for best Python 3.12 + NVIDIA performance.  
> Deliver:  
> - Recommended replacements/options (e.g., `onnxruntime-gpu`, `torch-tensorrt`).  
> - Exact install commands for CUDA-enabled and CPU-only profiles.  
> - Trade-offs (speed, memory, portability, install complexity).  
> - Recommended default profile for local training and for CI.

### Prompt D — Stale guidance scanner

> Scan docs/scripts for stale statements that conflict with current requirements (Python version, torch/tensorflow versions, CUDA instructions).  
> Update inconsistencies and provide a before/after change log with file+line references.

### Prompt E — Hard validation gate before merge

> Before finalizing:  
> - run dependency install checks  
> - run `pip check`  
> - run preflight checker  
> - run GPU verification script  
> - run one short training smoke test  
> If any command fails, fix it or mark it as environment-limited and rerun what is possible.

---

## 4A) Immediate Next Commands (Phase C unblock in real 3.12 env)

Run these in your Python 3.12 machine to complete Phase C:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip check
python -c "mods=['tensorflow','torch','torchvision','ccxt','pandas','numpy','PyQt5','web3']; [(__import__(m), print('OK', m)) for m in mods]"
python nexlify_preflight_checker.py --symbol BTC/USDT --automated
python scripts/verify_gpu_training.py --quick
python scripts/phase_c_validation.py
# Optional strict mode (non-zero on WARN/FAIL):
python scripts/phase_c_validation.py --strict
# Optional JSON report artifact (includes `overall_status` + per-check `check_id`):
python scripts/phase_c_validation.py --json-out phase_c_report.json
# Optional Markdown report artifact:
python scripts/phase_c_validation.py --md-out phase_c_report.md
# Optional: force heavy checks even when imports are missing
python scripts/phase_c_validation.py --force-heavy
# Optional: shorten verbose command output
python scripts/phase_c_validation.py --max-detail-lines 40
# Optional: summary-only console output (useful in CI logs)
python scripts/phase_c_validation.py --summary-only
# Optional: enforce failure on SKIP states (for strict CI gates)
python scripts/phase_c_validation.py --strict --fail-on-skip
# Optional: validate Phase C validator logic tests
python -m unittest tests/test_phase_c_validation_unittest.py
```

## 4B) Phase C Completion Handoff (copy/paste checklist)

Run this on a real Python 3.12 machine and mark each item as you complete it:

- [ ] `python3.12 -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -U pip && pip install -r requirements.txt`
- [ ] `python scripts/phase_c_validation.py --summary-only --command-timeout 180`
- [ ] `python scripts/phase_c_validation.py --strict --fail-on-skip --command-timeout 180`
- [ ] `python nexlify_preflight_checker.py --symbol BTC/USDT --automated`
- [ ] `python scripts/verify_gpu_training.py --quick`
- [ ] `python -m unittest tests/test_phase_c_validation_unittest.py`

**Phase C exit criteria:**
- `phase_c_validation.py --strict --fail-on-skip` exits with code `0`.
- Import probe has no missing modules (`tensorflow`, `torch`, `torchvision`, `ccxt`, `pandas`, `numpy`, `PyQt5`, `web3`).
- Preflight and GPU quick verification both return PASS in validator output.
- At least one short training smoke test completes without dependency/runtime errors.

## 4C) Phase D Kickoff Handoff (NVIDIA optimization pass)

Run these on a GPU-capable Python 3.12 host:

```bash
# 1) Confirm NVIDIA stack visibility
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda)"

# 2) Run project GPU verification
python scripts/verify_gpu_training.py --quick

# 3) Run full Phase C validator with heavy checks forced
python scripts/phase_c_validation.py --force-heavy --strict --fail-on-skip --command-timeout 300
```

**Phase D minimum kickoff criteria:**
- NVIDIA driver + CUDA runtime are visible (`nvidia-smi` works).
- `torch.cuda.is_available()` returns `True` on target GPU host.
- `scripts/verify_gpu_training.py --quick` passes GPU detection and optimization profile checks.

## 4D) Troubleshooting Matrix (Python/CUDA/Wheels)

| Symptom | Likely Cause | Quick Check | Resolution |
|---|---|---|---|
| `python` check warns `needs 3.12+` in Phase C | Interpreter is not Python 3.12 | `python -V` | Create/activate a 3.12 venv and rerun validation. |
| `No module named torch/tensorflow/ccxt/...` | Dependencies not installed in active environment | `pip show torch tensorflow ccxt` | `pip install -r requirements.txt` in the active venv. |
| `torch.cuda.is_available() == False` on GPU host | Driver/CUDA/runtime mismatch or CPU-only wheel | `nvidia-smi` and `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` | Install compatible NVIDIA driver/CUDA and a CUDA-enabled torch build. |
| `nvidia-smi` not found | NVIDIA driver utilities absent/not on PATH | `which nvidia-smi` | Install/repair NVIDIA driver tooling and ensure PATH is configured. |
| `pip check` reports broken requirements | Conflicting package versions | `pip check` | Reinstall from pinned requirements in a clean 3.12 venv. |
| `Preflight`/`GPU verify` show SKIP in validator | Import probe missing required modules | `python scripts/phase_c_validation.py --summary-only` | Install dependencies, then rerun with `--force-heavy`. |

## 4E) Phase F Release-Readiness Handoff

Use this on your target Python 3.12 environment after Phase C/D checks pass.

### 1) Capture known-good environment tuple
```bash
python -V
pip -V
python -c "import platform; print('platform=', platform.platform())"
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'cuda_available=', torch.cuda.is_available())"
python -c "import tensorflow as tf; print('tensorflow=', tf.__version__)"
python -c "import ccxt, sqlalchemy, numpy; print('ccxt=', ccxt.__version__, 'sqlalchemy=', sqlalchemy.__version__, 'numpy=', numpy.__version__)"
```

### 2) Freeze the tested environment
```bash
pip check
pip freeze > requirements-lock-$(python -c "import datetime; print(datetime.date.today().isoformat())").txt
```

### 3) Verify release gates
```bash
python scripts/phase_c_validation.py --force-heavy --strict --fail-on-skip --command-timeout 300
python scripts/verify_gpu_training.py --quick
python -m unittest tests/test_phase_c_validation_unittest.py
```

### 4) Record final migration status
- [ ] Attach validator JSON/Markdown reports to release notes.
- [ ] Record known-good tuple (Python/CUDA/driver/torch/tensorflow/ccxt/sqlalchemy/numpy).
- [ ] Commit lockfile or constraints strategy decision.
- [ ] Mark Phase C/D/F complete in this playbook.

## 5) Progress Log

> Update this section continuously as migration work proceeds.

### 2026-03-13 (initial draft)
- Created living migration playbook with phased plan + reusable prompts.
- Next action: execute **Phase A** and **Phase B** in a focused PR.

### 2026-03-13 (phase A/B started)
- Updated packaging metadata baseline to Python 3.12+ (`python_requires`, classifiers).
- Updated primary runtime/version guard scripts from 3.11+ to 3.12+.
- Aligned key docs and setup guides to Python 3.12+ messaging.

### 2026-03-13 (consistency pass #2)
- Updated top-level README Python badge to 3.12+.
- Updated `TRAINING_GUIDE.md` prerequisite baseline to Python 3.12+.
- Refreshed baseline snapshot notes to reflect completed script guard migration progress.

### 2026-03-13 (consistency pass #3)
- Updated `nexlify_preflight_checker.py` troubleshooting guidance from Python 3.8+ to 3.12+.
- Re-ran repository scan for stale Python baseline references in core docs/scripts.

### 2026-03-13 (validation pass #1)
- Ran `pip check` in the current environment: no broken requirements found.
- Normalized lingering dependency comment language to align with Python 3.12 baseline.

### 2026-03-13 (phase-tracking update)
- Added a phase status dashboard (A–F) so progress is visible at a glance.
- Marked Phase A/B checklist items complete and reflected Phase C partial progress.
- Recorded current blockers for Phase C in this container (Python 3.10 runtime + missing ML dependencies).

### 2026-03-13 (phase E cleanup #1)
- Updated stale PyTorch/CUDA version guidance in GPU docs to align with the Python 3.12 + torch 2.6.0 baseline.
- Refreshed `docs/PYTORCH_VERSION_NOTES.md`, `docs/GPU_SETUP.md`, `docs/GPU_TRAINING_GUIDE.md`, `docs/IMPLEMENTATION_SUMMARY.md`, and `docs/MULTI_MODE_RL_TRAINING.md`.
- Phase status: **E remains in progress** (more long-tail docs may still need cleanup).

### 2026-03-13 (phase E cleanup #2)
- Updated remaining stale PyTorch/CUDA footer and requirements notes in `docs/GPU_TRAINING_GUIDE.md` and `docs/ADAPTIVE_RL_GUIDE.md`.
- Continued long-tail documentation cleanup for version consistency.

### 2026-03-13 (phase E cleanup #3)
- Updated `CLAUDE.md` and `ULTRA_OPTIMIZED_SYSTEM.md` to align key version guidance with current Python 3.12 / requirements baseline.
- Added historical-snapshot disclaimers to `VALIDATION_REPORT.md` and `CODEBASE_AUDIT_REPORT.md` to avoid version confusion.
- Phase status: **E still in progress**, but baseline-vs-historical guidance is now clearer.

### 2026-03-13 (phase E cleanup #4)
- Completed a non-historical stale-version sweep; no additional active baseline mismatches found outside archived snapshot reports.
- Added an explicit **Immediate Next Commands** block to accelerate Phase C completion in a real Python 3.12 environment.
- Revalidated key Python script syntax checks (`py_compile`) after recent migration edits.

### 2026-03-13 (phase C validation pass #2)
- Re-ran Phase C checks in current container: `python --version`, `pip check`, import smoke probe, preflight, and GPU quick verification.
- Findings: environment remains Python 3.10.19 with core ML/trading dependencies missing, so Phase C runtime validation is still blocked locally.
- Phase status: **C remains partial**, with executable completion commands documented in section 4A.

### 2026-03-13 (phase C tooling #1)
- Added `scripts/phase_c_validation.py` to standardize Phase C checks (Python version, `pip check`, import probe, preflight, GPU quick verify).
- Updated section 4A commands to include the new validator script.
- Phase status: **C still partial** in this container, but validation is now reproducible with one command.

### 2026-03-13 (phase C tooling #2)
- Enhanced `scripts/phase_c_validation.py` with `--strict` mode and optional `--json-out` report output.
- Added environment-aware warning classification so dependency-missing/container constraints are reported as WARN instead of hard FAIL in default mode.
- Re-ran validator in default and strict modes to confirm behavior.

### 2026-03-13 (phase C tooling #3)
- Updated `scripts/phase_c_validation.py` to emit `SKIP` for heavy checks when required imports are missing (unless `--force-heavy` is used).
- Added summary counts (PASS/WARN/FAIL/SKIP) and retained strict-mode behavior for CI-style enforcement.
- Updated section 4A with `--force-heavy` usage.

### 2026-03-13 (phase C tooling #4)
- Improved `scripts/phase_c_validation.py` output ergonomics with `--max-detail-lines` truncation control.
- Extended JSON report format to include top-level summary counts and run metadata (`strict_mode`, `force_heavy`, `python`).
- Kept default behavior focused on actionable migration signals while preserving strict enforcement options.

### 2026-03-13 (phase C tooling #5)
- Added Markdown report output support to `scripts/phase_c_validation.py` via `--md-out` for easier sharing in issues/PRs.
- Kept JSON output for machine-readable automation and added report metadata consistency with summary counts.
- Updated section 4A with Markdown artifact command example.

### 2026-03-13 (phase C tooling #6)
- Added `--summary-only` mode to `scripts/phase_c_validation.py` for compact console output in CI/log-heavy runs.
- Updated section 4A with a summary-only command example.
- Phase status: **C remains partial** in this container, but validator ergonomics for iterative runs improved.

### 2026-03-13 (phase C tooling #7)
- Added actionable recommendation output to `scripts/phase_c_validation.py` so each run suggests concrete next commands based on WARN/SKIP states.
- Keeps summary-only mode concise while still providing migration guidance after each run.

### 2026-03-13 (phase C tooling #8)
- Added `overall_status` and stable `check_id` fields to Phase C JSON/MD reports for easier automation and dashboarding.
- Updated validator summary output to print overall status explicitly.
- Updated section 4A notes to reflect richer report metadata.

### 2026-03-13 (phase C tooling #9)
- Added `--fail-on-skip` mode to `scripts/phase_c_validation.py` for stricter CI gating when SKIP states should fail builds.
- Extended JSON/Markdown report payloads with `recommended_next_steps` and `fail_on_skip` metadata.
- Updated section 4A with strict CI gating command example.

### 2026-03-13 (phase C tooling #10)
- Added explicit exit-policy reporting to `scripts/phase_c_validation.py` (console + JSON + Markdown) so pass/fail decisions are auditable.
- Reports now include `exit_policy` and `exit_code` metadata for CI/debug clarity.
- Phase status: **C still partial** in this container, but validator outputs are now policy-transparent.

### 2026-03-13 (phase C tooling #11)
- Added stdlib unit tests for `scripts/phase_c_validation.py` decision logic (`determine_overall_status`, `compute_exit_code`, `get_recommendations`).
- Added optional command in section 4A to run validator logic tests.
- Phase status: **C still partial** in this container, but validator behavior now has regression coverage.

### 2026-03-14 (doc consistency pass + status cadence)
- Updated `README.md` prerequisites to explicitly require Python 3.12+ (removed lingering 3.11 phrasing).
- Updated `docs/IMPLEMENTATION_GUIDE.md` system requirements to Python 3.12+ (removed lingering 3.9 phrasing).
- Confirmed ongoing status-tracking cadence with a reusable checklist format per progress update.
- Phase status: **C still partial** in this container; **E remains in progress** while long-tail doc consistency cleanup continues.

### 2026-03-14 (phase C handoff pack)
- Added a copy/paste **Phase C Completion Handoff** checklist (section 4B) for running final validation in a real Python 3.12 environment.
- Added explicit Phase C exit criteria so completion is unambiguous (`--strict --fail-on-skip` must exit 0, imports present, preflight/GPU checks pass, smoke test passes).
- Re-ran local validator in this container to reconfirm environment-limited status (WARN/SKIP expected here).

### 2026-03-14 (phase D kickoff prep)
- Started Phase D as **in progress (container-blocked)** and documented a dedicated 4C handoff checklist for NVIDIA optimization pass execution on a GPU-capable Python 3.12 host.
- Recorded local container probe results: `nvidia-smi` unavailable and `scripts/verify_gpu_training.py --quick` fails due to missing heavy deps (`torch`, `numpy`, `ccxt`) in this environment.
- Locked default profile decision in checklist as NVIDIA-local primary with CPU/CI fallback.

### 2026-03-14 (phase E cleanup #5)
- Updated active docs to replace legacy `pynvml` install guidance with `nvidia-ml-py` in `ULTRA_OPTIMIZED_SYSTEM.md`.
- Added a reusable troubleshooting matrix (section 4D) covering Python-version mismatch, CUDA/driver mismatch, wheel mismatch, and SKIP-state recovery.
- Marked the Phase E troubleshooting-matrix checklist item complete.

### 2026-03-14 (phase E cleanup #6)
- Completed active-document stale-version sweep for Python/torch/tensorflow/CUDA references; no remaining baseline mismatches found in active docs/scripts.
- Marked Phase E checklist item for stale-version reference removal complete.
- Phase status updated: **E complete (active docs)**; historical snapshot reports remain intentionally versioned and are guarded by historical-note disclaimers.

### 2026-03-14 (phase F prep #1)
- Started Phase F as **in progress (handoff prepared)**.
- Added section 4E with a release-readiness handoff: environment tuple capture, freeze workflow, release gates, and final closeout checklist.
- Updated Phase F checklist language to explicitly require recording a constraints/lock strategy decision.

---

## 6) Open Questions — Best-Practice Decisions

### Q1) Keep mixed-platform notes (MPS/ROCm) or trim to NVIDIA-first?
**Recommendation:** Keep a **NVIDIA-first main path**, but retain a short appendix for MPS/ROCm.

**Best practice:**
- Main install/training docs should optimize for one primary path (NVIDIA) to reduce ambiguity.
- Keep non-primary platforms in a compact “Alternative Backends” section to avoid fragmentation.
- Mark each path with support level: `Primary (NVIDIA)`, `Secondary (MPS/ROCm)`, `Experimental`.

**Decision for this project:**
- Use NVIDIA as the default path in all top-level guides.
- Keep MPS/ROCm notes, but move them to clearly scoped subsections.

### Q2) Separate `requirements-gpu.txt` and `requirements-cpu.txt`?
**Recommendation:** Yes—split profiles, plus a shared base file.

**Best practice layout:**
- `requirements/base.txt` → common packages used everywhere
- `requirements/gpu-nvidia.txt` → base + CUDA/NVIDIA extras
- `requirements/cpu.txt` → base + CPU-friendly alternatives
- optional: `requirements/dev.txt` for lint/test/tooling

**Why:**
- Faster CI installs and fewer heavy GPU dependency failures in non-GPU environments.
- Clearer reproducibility of runtime intent (GPU vs CPU).
- Easier troubleshooting and smaller blast radius for dependency updates.

### Q3) Generate a lockfile?
**Recommendation:** Yes, for deterministic rebuilds.

**Best practice:**
- Keep human-maintained top-level requirement files (`base/gpu/cpu`).
- Generate pinned lock artifacts per profile (e.g., with `pip-tools`):
  - `requirements/locks/gpu-nvidia-py312.txt`
  - `requirements/locks/cpu-py312.txt`
- Regenerate locks only when intentionally upgrading dependencies.
- In automation, install from lockfiles; in development, edit high-level files.

### Immediate Follow-up Actions
- [ ] Create requirements profile split (`base`, `gpu-nvidia`, `cpu`, `dev`).
- [ ] Add lockfile generation workflow (e.g., `pip-compile`) for Python 3.12.
- [ ] Update docs to present NVIDIA-first flow and link alternative backend appendix.
