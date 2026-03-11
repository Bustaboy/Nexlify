# Nexlify Remediation Plan: Fix Docs/UI Drift and Implement Missing Items

Date: 2026-03-11
Owner: Engineering
Status: Ready for execution

## 1) Goal
Close all high-impact gaps identified in `docs/PROJECT_STATE_VERIFICATION.md` by:

1. Making launch/entry points accurate and runnable.
2. Aligning documentation with actual package paths and GUI structure.
3. Ensuring advanced feature tabs are visible and testable in UI flows.
4. Adding automated checks so doc/code drift does not recur.

## 2) Missing / Mismatched Items to Fix

### A. Entry-point mismatch
- Docs reference `python nexlify_launcher.py`, but the root launcher file is missing.
- Action: implement `nexlify_launcher.py` (or update docs to canonical launcher only).

### B. Architecture/module naming drift
- Docs reference legacy/root module names (e.g., `cyber_gui.py`, `nexlify_tax_reporting.py`) inconsistent with package layout.
- Action: normalize all docs to real package paths (`nexlify/gui/cyber_gui.py`, `nexlify/financial/nexlify_tax_reporter.py`, etc.).

### C. UI tab naming mismatch
- README tab names do not match top-level `CyberGUI` tabs.
- Action: either:
  - update README labels to current UI, OR
  - add alias text/help overlays in UI if legacy terms must be preserved.

### D. Runtime verification gaps
- Current environment may miss runtime deps; test and GUI smoke checks can fail at import time.
- Action: add deterministic preflight checks and CI jobs to separate dependency failures from product failures.

### E. No automated docs-vs-code guard
- Action: create `scripts/verify_docs_vs_code.py` and wire into CI.

## 3) Execution Plan (Phased)

### Phase 0 — Baseline + branch hygiene
- [x] Re-run project inventory and capture current truth table (docs claim vs code location).
- [x] Create issue checklist and map each mismatch to a concrete file change.

### Phase 1 — Implement missing runnable entry point
- [x] Add `nexlify_launcher.py` with:
  - dependency preflight
  - config path resolution
  - launch handoff to `nexlify.gui.cyber_gui`
  - clear logging + non-zero exits on failure
- [x] Add/adjust test for launcher import and CLI smoke behavior.

### Phase 2 — Documentation correction pass
- [x] Update `README.md` launch instructions and architecture table.
- [x] Update docs that mention stale filenames/old tabs.
- [x] Add one “Source of Truth” section listing canonical entry points and UI tabs.

### Phase 3 — UI discoverability alignment
- [x] Confirm advanced tabs (Emergency/Tax/DeFi/Withdrawals) are integrated in the main GUI flow.
- [x] Add small UX copy updates where needed so labels match docs.
- [x] Take screenshots of perceptible UI changes. (N/A: this remediation pass did not change rendered UI components)

### Phase 4 — Automation and CI guardrails
- [x] Implement `scripts/verify_docs_vs_code.py` checks:
  - referenced file exists
  - referenced module importable
  - documented tab labels appear in code (or mapped aliases)
- [x] Add CI job to run verification script and fail on drift.

### Phase 5 — Validation + release notes
- [x] Run targeted tests + smoke checks.
- [x] Add final verification report with pass/fail matrix.
- [x] Update changelog/release note section.

## 4) Acceptance Criteria
- `python nexlify_launcher.py` works (or docs no longer claim it).
- No stale module/file names remain in core docs.
- README GUI tab section reflects actual UI labels and location.
- Advanced feature tabs are reachable from primary GUI.
- CI includes docs-vs-code verification and passes.

## 5) Self-Prompts (Execution Prompts for Agent Runs)

Use these prompts in sequence for focused implementation sessions.

### Prompt 1: Entry point implementation
"Audit launch paths and implement any missing launcher entry point so README launch commands are valid. Add minimal tests and update docs if command semantics change."

### Prompt 2: Docs normalization
"Scan README + docs for stale module/file references and replace with current canonical package paths. Produce a table mapping old names to new names in docs."

### Prompt 3: UI labels and navigation consistency
"Compare documented GUI tabs with actual `CyberGUI` tabs. Update docs (and UI labels only if necessary) so users can locate each feature quickly, including where API config and advanced tabs live."

### Prompt 4: Automation script
"Create `scripts/verify_docs_vs_code.py` to validate documented entrypoints/modules/tabs exist in code. Add it to CI and provide actionable failure messages."

### Prompt 5: Runtime validation
"Run targeted tests and headless GUI smoke checks. If deps are missing, clearly separate environment failures from code failures and record reproducible commands."

### Prompt 6: Final QA + PR
"Summarize all changes with before/after evidence, include screenshots for UI-visible updates, list executed commands with outcomes, and ensure docs and CI checks pass."

## 6) Risk Controls
- Keep remediation changes split into small commits by phase.
- Avoid broad refactors while correcting docs/entry points.
- For each renamed reference, preserve backward-compat links where feasible.

## 7) Deliverables
1. Fixed launcher and/or corrected launch docs.
2. Updated docs with canonical module paths and tab names.
3. Optional UI copy tweaks for consistency.
4. Verification script + CI integration.
5. Final validation report.
