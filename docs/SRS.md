---
schema: cochem-srs/5.0
repo: CoChem-GEOM
status: ACTIVE_CORE
pipeline_baseline: 5.0.0
source_commit: 2a0c1ccae5375b6b6231cec118cda23652473408
decision_ref: swarm-1126-8efd
---

# CoChem-GEOM — Canonical Software Requirements Specification (Pipeline 5.0.0)

## 1. Purpose, Scope & System Classification

`CoChem-GEOM` is an `ACTIVE_CORE` repository in the CoChem Pipeline 5.0.0 ecosystem providing rotational spectroscopy molecular geometry fitting (weighted nonlinear least-squares $r_0$, honest Costain/Kraitchman substitution coordinates $r_s$ with imaginary-coordinate handling), Pickett `.var`/`.par`/`.fit`/`.lin` deck ingestion, and converged canonical CCSD equilibrium structure (`r_e`) benchmarking via ORCA 6.1.1 and PySCF.

- **Canonical shipped package (`pyproject.toml`):** `src/cochem_geom/` with explicit zero-wildcard package allowlist `["cochem_geom", "cochem_geom.engine", "cochem_geom.interfaces", "cochem_geom.literature"]` (`sys-1126-a3d4`).
- **CLI entrypoint:** `cochem-geom` (`src/cochem_geom/cli.py:main`) providing `parse-pickett`, `benchmark-h2o`, and `bundle-hcooh-dme`.
- **Ecosystem module provider:** `cochem.modules` entrypoint `geom = cochem_geom.provider:geom_provider` (`src/cochem_geom/provider.py`).
- **Authoritative shipped modules:** `src/cochem_geom/pickett.py`, `src/cochem_geom/constants.py`, `src/cochem_geom/inertia.py`, `src/cochem_geom/kraitchman.py`, `src/cochem_geom/r0_fitter.py`, `src/cochem_geom/ccsd_driver.py`, `src/cochem_geom/reporting.py`, `src/cochem_geom/engine/constraints.py`, `src/cochem_geom/interfaces/vibrations.py`, and `src/cochem_geom/literature/peptide_mapping.py`.

## 2. Pipeline 5.0.0 Architecture Alignment & Resolved Conflicts

`CoChem-GEOM` aligns with Pipeline 5.0.0 governance (`swarm-1126-8efd`, `sys-1126-a3d4`):

- **CONTRA-01 (Execution Topology):** Local WSL2 workstation (`~/CoChem_Workstation_Direct/Modules/geom-env`) and Cloud Run / Colab cloud offload (`cochem-mcp-mailbox-request/2`) are the authoritative execution lanes for heavy CCSD/VPT2 jobs; legacy Codespaces-first assumptions are retired.
- **CONTRA-02 & CONTRA-03 (Scheduling & Job Harness):** Long-running CCSD and anharmonic VPT2 continuation workloads (such as the 5-isotopologue HCOOH-DME Rung-2 continuation manifest in `benchmarks/hcooh_dme_rung2/pipeline_500_continuation_manifest.json`) route via `cochem-mcp-mailbox-request/2` and `cochem-scheduled-prompt/1` rather than retired `g01..g22` lanes.
- **CONTRA-06 (Dual-Tree & Namespace Consolidation):** `CoChem-GEOM` ships ONLY `src/cochem_geom/*` under an explicit zero-wildcard allowlist in `pyproject.toml` so it does not collide with `CoChem-BASE` or ship unverified numbered stage directories (`00-SETUP-AA/`..`04-REPORT-AA/`). All 14 tracked stage `.py` files are commit-pinned and dispositioned in Section 7 and `docs/archive/README.md`.

## 3. Authoritative Scientific Engines, Methods & Method Matrix Constraints

- **Masses & Physical Constants (`src/cochem_geom/constants.py`, `src/cochem_geom/inertia.py`):** Exact IUPAC/AME2020 isotopic masses and Groner conversion constant (`505379.0` MHz u A^2) with principal-axis-system (PAS) diagonalization and inertial defect checks.
- **Kraitchman $r_s$ & Weighted $r_0$ (`src/cochem_geom/kraitchman.py`, `src/cochem_geom/r0_fitter.py`):** Computes authentic Kraitchman substitution coordinates with Costain error propagation (`sigma_c = 0.0015 / |c|`), reporting near-zero/imaginary coordinates honestly without fabrication. Weighted Levenberg-Marquardt $r_0$ fitting propagates full Jacobian covariance.
- **Canonical CCSD Equilibrium Driver (`src/cochem_geom/ccsd_driver.py`):** Supports genuine ORCA 6.1.1 canonical CCSD (`NumGrad NumFreq`, `%mdci STol 1.0e-8`) and PySCF CPU CCSD/geomeTRIC optimization. Explicitly forbids `DLPNO-CCSD(T)` geometry optimization (`ValueError`) because ORCA MDCI lacks analytic DLPNO-CCSD(T) nuclear gradients.
- **Method Matrix §4.4 `%geom` & `! TightOpt` Constraint:** `generate_orca_ccsd_input()` pairs `! {ref} CCSD {basis} TightOpt VeryTightSCF NumGrad NumFreq` with all five Method Matrix §4.4 `%geom` convergence thresholds (`TolE 1.0e-7`, `TolRMSG 2.0e-6`, `TolMaxG 5.0e-6`, `TolRMSD 5.0e-5`, `TolMaxD 1.0e-4`, `EnForceStrictConvergence true`) — strictly tighter than §4.4 gradient limits (`3e-6`, `1e-5`) and verified fail-closed by `tests/test_tightopt_geom_gate.py`.

## 4. Cross-Repo Interfaces, Contracts & Canonical Schemas

- **`CoChem-BASE` Module Provider (`src/cochem_geom/provider.py`):** Registers `geom = cochem_geom.provider:geom_provider` exposing `parse_pickett`, `fit_weighted_r0`, `compute_kraitchman_rs`, and `optimize_ccsd` metadata and operations.
- **Pickett / `CoChem-SpycFit` Ingestion (`src/cochem_geom/pickett.py`):** Parses authentic Pickett `.var`, `.par`, `.fit`, and `.lin` files into structured rotational constants and uncertainties with SHA-256 provenance.
- **Separated Scientific Reporting (`src/cochem_geom/reporting.py`):** Emits separated JSON and LaTeX comparison tables explicitly distinguishing `r_e` (ORCA CCSD/cc-pVTZ), `r_e` (PySCF CCSD/cc-pVDZ), `r_s` (Kraitchman), `r_0` (weighted fit), and `r_e^SE` (explicitly `PENDING` whenever isotope-specific $\Delta B_{\text{vib}}$ corrections are absent; never conflates $r_0$ or $r_e$ with $r_e^{\text{SE}}$).
- **Verified Benchmarks:** `benchmarks/h2o_t_geom_01/` contains converged real ORCA 6.1.1 CCSD/cc-pVTZ and PySCF CCSD/cc-pVDZ H2O geometries validated against NIST CCCBDB (`tests/test_ccsd_and_reporting.py`); `benchmarks/hcooh_dme_rung2/` holds 5 parsed HCOOH-DME isotopologue decks with `ccsd_equilibrium_geometry_status = "PENDING"` and `re_se_status = "PENDING"`.

## 5. Verification, Zero-Mock CI Gates & Acceptance Criteria

`CoChem-GEOM` enforces fail-closed automated verification (`.github/workflows/zero_mock_ci.yml` and local `pytest`):

- **Zero-Mock & Anti-Spoof Gate (`tests/test_zero_mock_and_no_sim_gate.py`):** Zero banned simulation/mock patterns in production code and zero hallucinated water rotational constants (enforced across all tracked files by `tests/test_ccsd_and_reporting.py`).
- **Method Matrix `! TightOpt` / `%geom` Gate (`tests/test_tightopt_geom_gate.py`):** Enforces full §4.4 five-tolerance `%geom` compliance (`TolE <= 1e-7`, `TolRMSG <= 3e-6`, `TolMaxG <= 1e-5`, `TolRMSD <= 5e-5`, `TolMaxD <= 1e-4`), `EnForceStrictConvergence true`, `VeryTightSCF`, and `%mdci STol <= 1e-8`, while rejecting bare or under-tightened `! TightOpt` decks.
- **Core Scientific Test Suites:** `tests/test_pickett_parser.py`, `tests/test_r0_and_kraitchman.py`, `tests/test_ccsd_and_reporting.py`, `tests/test_symmetry_warning.py`, and `tests/test_geom_pipeline.py` (all executed with `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider tests`).

## 6. Verified Implementation Gaps, Known Holds & Non-Goals

Per the Pipeline 5.0.0 core repository audit, the following capabilities are honestly documented as unimplemented or pending:

- **Semi-Experimental Equilibrium ($r_e^{\text{SE}}$), $r_z$, and Costain-Laurie $r_m^{(1)}/r_m^{(2)}$:** Not yet implemented in `src/cochem_geom/`; `r_e_se` is emitted strictly as `PENDING` until authentic isotope-specific $\Delta B_{\text{vib}}$ anharmonic corrections are supplied by `CoChem-TORQ`/ORCA/CFOUR.
- **Generic User-Molecule CLI:** `src/cochem_geom/cli.py` currently exposes `parse-pickett`, `benchmark-h2o`, and `bundle-hcooh-dme`; generic CLI subcommands for arbitrary user-molecule $r_0$/$r_s$/CCSD runs remain planned work.
- **Unimplemented v4.0 Blueprint Engines:** GPU-accelerated (`PyTorch`/`CuPy`) multi-seed fitting, SciPy sparse Hessian inversion (`> 50` atoms), Bayesian MCMC error propagation, Diagonal Born-Oppenheimer Corrections (DBOC), and automated isotopologue graph branching are not implemented.
- **Unshipped Stage Scripts (`00-SETUP-AA/`..`04-REPORT-AA/` & `04-ANALYSIS/`):** Six stage scripts (`cochem_geom_eckart.py`, `cochem_geom_eval_sym.py`, `cochem_geom_eval_triage.py`, `cochem_geom_fitter_core.py`, `cochem_geom_lms.py`, `cochem_geom_distance_hash.py`, plus `cochem_geom_reporter_ui.py` — 7 total including `04-ANALYSIS/`) remain outside `src/cochem_geom/` and are classified `NEEDS_INTEGRATION` prior to wheel inclusion. `src/cochem_geom/interfaces/vibrations.py` retains a minimal `MatplotlibVibrationalPlotter.__init__` stub.
- **Non-Goals:** `CoChem-GEOM` never approximates `DLPNO-CCSD(T)` optimizations in ORCA, never substitutes $r_0$ for $r_e^{\text{SE}}$, and never silently fabricates missing electronic $g$-tensor or $\Delta B_{\text{vib}}$ corrections.

## 7. Legacy Specification Disposition (Non-Normative Archive)

All root v4.0 blueprint and stage markdown specifications (`Architecture Mode_ Final CoChem-GEOM (v4.0) Master Blueprint.md`, `CoChem_GEOM_Workflow_Plan_v4.md`, `Stage 0.0 Master Setup.md` through `Stage 4.1 Reporter UI.md`, `Mol_Struct_Fit_Min.md`, `Molecular Structure Fitting Research.md`) are superseded by this Pipeline 5.0.0 `docs/SRS.md` and cataloged as **NON-NORMATIVE ARCHIVE** in `docs/archive/README.md`.

### Commit-Pinned Per-File Disposition of v4.0 Stage Python Modules (`2a0c1ccae5`)

| Tracked File | Disposition | Grounded Evidence |
| --- | --- | --- |
| `00-SETUP-AA/cochem_geom_setup.py` | `LEGACY_ARCHIVE` | v4.0 SWMR HDF5/CUDA bootstrap superseded by `src/cochem_geom/cli.py` & workstation env |
| `01-INGEST-AA/cochem_geom_eckart.py` | `NEEDS_INTEGRATION` | Standalone Eckart alignment & decoupling math; tested by `tests/test_geom_pipeline.py` |
| `01-INGEST-AA/cochem_geom_ingest_math.py` | `PORTED_TO_SRC` | Wrapper delegating to `src/cochem_geom/constants.py` & `src/cochem_geom/inertia.py` |
| `01-INGEST-AA/cochem_geom_ingest_parser.py` | `PORTED_TO_SRC` | Wrapper delegating to `src/cochem_geom/pickett.py` |
| `02-EVAL-AA/cochem_geom_eval_sym.py` | `NEEDS_INTEGRATION` | MolSym HITL symmetry controller widget not yet ported into `src/cochem_geom/` |
| `02-EVAL-AA/cochem_geom_eval_triage.py` | `NEEDS_INTEGRATION` | DoF variable-reduction & correlation-threshold widget not yet ported into `src/cochem_geom/` |
| `03-FIT-AA/cochem_geom_fitter_core.py` | `NEEDS_INTEGRATION` | Z-matrix & quaternion internal-coordinate engine not yet ported into `src/cochem_geom/r0_fitter.py` |
| `03-FIT-AA/cochem_geom_fitter_optim.py` | `PORTED_TO_SRC` | Wrapper delegating to `src/cochem_geom/kraitchman.py` & `src/cochem_geom/r0_fitter.py` |
| `03-FIT-AA/cochem_geom_lms.py` | `NEEDS_INTEGRATION` | Low-Mode Sampling generator with flat `sys.path` imports; requires port to `src/cochem_geom/` |
| `03-FIT-AA/cochem_geom_router.py` | `LEGACY_ARCHIVE` | v4.0 3-tier routing stub superseded by `src/cochem_geom/ccsd_driver.py` & Pipeline 5.0.0 runner |
| `03b-REFINE-AA/cochem_geom_ccsdt_refine.py` | `PORTED_TO_SRC` | Wrapper delegating directly to `src/cochem_geom/ccsd_driver.py` |
| `04-ANALYSIS/cochem_geom_distance_hash.py` | `NEEDS_INTEGRATION` | Distance-matrix conformer deduplication hasher required by `cochem_geom_lms.py` |
| `04-REPORT-AA/cochem_geom_reporter_latex.py` | `PORTED_TO_SRC` | Wrapper delegating directly to `src/cochem_geom/reporting.py` |
| `04-REPORT-AA/cochem_geom_reporter_ui.py` | `NEEDS_INTEGRATION` | Interactive Plotly/py3Dmol displacement UI not yet ported into `src/cochem_geom/reporting.py` |
