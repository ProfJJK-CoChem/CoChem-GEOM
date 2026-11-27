# CoChem-GEOM Legacy Specification & Stage Script Archive Index

> **NON-NORMATIVE ARCHIVE:** All root v4.0 blueprint documents and stage markdown specifications listed below are historical pre-v5.0.0 design baselines retained in place so existing repository tests and historical links remain intact. The single authoritative Pipeline 5.0.0 specification is [`docs/SRS.md`](../SRS.md) (ratified under `swarm-1126-8efd` and `sys-1126-a3d4`).

## Non-Normative Root Markdown Documents

| Root Document | Historical Role | Disposition |
| --- | --- | --- |
| `Architecture Mode_ Final CoChem-GEOM (v4.0) Master Blueprint.md` | v4.0 architectural blueprint | **NON-NORMATIVE ARCHIVE** |
| `CoChem_GEOM_Workflow_Plan_v4.md` | v4.0 stage workflow plan | **NON-NORMATIVE ARCHIVE** |
| `Mol_Struct_Fit_Min.md` | Historical structure-fitting notes | **NON-NORMATIVE ARCHIVE** |
| `Molecular Structure Fitting Research.md` | Historical literature & methodology notes | **NON-NORMATIVE ARCHIVE** |
| `Stage 0.0 Master Setup.md` | v4.0 Stage 0.0 specification | **NON-NORMATIVE ARCHIVE** |
| `Stage 1.0 Pre-Flight Parser.md` | v4.0 Stage 1.0 specification | **NON-NORMATIVE ARCHIVE** |
| `Stage 1.1 Ingestion Math.md` | v4.0 Stage 1.1 specification | **NON-NORMATIVE ARCHIVE** |
| `Stage 2.1 Eval Sym.md` | v4.0 Stage 2.1 specification | **NON-NORMATIVE ARCHIVE** |
| `Stage 2.2 Eval Triage.md` | v4.0 Stage 2.2 specification | **NON-NORMATIVE ARCHIVE** |
| `Stage 3.1 Fitter Core.md` | v4.0 Stage 3.1 specification | **NON-NORMATIVE ARCHIVE** |
| `Stage 3.2 Fitter Optim.md` | v4.0 Stage 3.2 specification | **NON-NORMATIVE ARCHIVE** |
| `Stage 3.3 CCSDT Refine.md` | v4.0 Stage 3.3 specification | **NON-NORMATIVE ARCHIVE** |
| `Stage 4.0 Reporter Latex.md` | v4.0 Stage 4.0 specification | **NON-NORMATIVE ARCHIVE** |
| `Stage 4.1 Reporter UI.md` | v4.0 Stage 4.1 specification | **NON-NORMATIVE ARCHIVE** |

## Commit-Pinned Disposition of Unshipped Stage Python Directories (`2a0c1ccae5`)

`pyproject.toml` builds exclusively from `src/cochem_geom/` (explicit allowlist: `cochem_geom`, `cochem_geom.engine`, `cochem_geom.interfaces`, `cochem_geom.literature`). The 14 tracked `.py` files across `00-SETUP-AA/`..`04-REPORT-AA/` and `04-ANALYSIS/` are NOT shipped in distribution wheels and carry the following commit-pinned dispositions (full SHA-256 manifest recorded in `sys_a3d4_torq_geom_wheel_diff.json`):

- **`PORTED_TO_SRC` (5 files):** `01-INGEST-AA/cochem_geom_ingest_math.py`, `01-INGEST-AA/cochem_geom_ingest_parser.py`, `03-FIT-AA/cochem_geom_fitter_optim.py`, `03b-REFINE-AA/cochem_geom_ccsdt_refine.py`, `04-REPORT-AA/cochem_geom_reporter_latex.py`.
- **`LEGACY_ARCHIVE` (2 files):** `00-SETUP-AA/cochem_geom_setup.py`, `03-FIT-AA/cochem_geom_router.py`.
- **`NEEDS_INTEGRATION` (7 files; 6 in `*-AA` + 1 in `04-ANALYSIS`):** `01-INGEST-AA/cochem_geom_eckart.py`, `02-EVAL-AA/cochem_geom_eval_sym.py`, `02-EVAL-AA/cochem_geom_eval_triage.py`, `03-FIT-AA/cochem_geom_fitter_core.py`, `03-FIT-AA/cochem_geom_lms.py`, `04-ANALYSIS/cochem_geom_distance_hash.py`, `04-REPORT-AA/cochem_geom_reporter_ui.py`.
