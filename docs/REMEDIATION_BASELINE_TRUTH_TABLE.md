# Remediation Baseline Truth Table (Docs Claim vs Code Reality)

Date: 2026-03-11
Purpose: Phase 0 baseline inventory for remediation execution.

| Area | Documentation Claim | Code Reality | Status | Action |
|---|---|---|---|---|
| Launcher entrypoint | `python nexlify_launcher.py` | Root `nexlify_launcher.py` now exists and launches GUI after preflight | ✅ Aligned | Keep verified in CI drift guard |
| Main GUI path | Legacy `cyber_gui.py` root references in docs | Canonical module path: `nexlify/gui/cyber_gui.py` | ✅ Aligned | Keep canonical path in docs |
| Tax module name | `nexlify_tax_reporting.py` in docs | Actual file: `nexlify/financial/nexlify_tax_reporter.py` | ✅ Aligned | Guard against stale string in README |
| Top-level GUI tabs | `Active Pairs`, `Profit Chart`, `Environment`, `API Config` | Top-level tabs are `Dashboard`, `Trading`, `Portfolio`, `Strategies`, `Settings`, `Logs` | ✅ Aligned | Keep README synced via drift checks |
| API Config location | Implied top-level API tab | API config is under `Settings` in `CyberGUI` | ✅ Aligned | Keep docs explicit on this location |
| Advanced tabs availability | Emergency/Tax/DeFi/Withdrawals described as features | Integrated via Phase 1/2 integration in UI integration module | ✅ Implemented | Verify via code checks in drift guard |
| Runtime validation confidence | Full UI working status implied by feature docs | Full runtime validation depends on local deps (`PyQt5`, `qasync`, etc.) | ⚠️ Environment-limited | Keep launcher preflight + report env constraints |

## Notes
- This table is the issue checklist mapping for remediation and traceability.
- It is intentionally concise and references canonical paths used in this repository.
