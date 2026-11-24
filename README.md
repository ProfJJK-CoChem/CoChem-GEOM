# CoChem-GEOM: FTMW Rotational Spectroscopy Structural Fitting & CCSD Comparison

**Author/PI:** Dr. Joshua John Klaassen  
**ORCiD:** [https://orcid.org/0009-0007-1506-4401](https://orcid.org/0009-0007-1506-4401)  
**GitHub Organization:** [https://github.com/ProfJJK-CoChem](https://github.com/ProfJJK-CoChem)  
**Experimental FTMW Collaborator:** Tennessee Tech University  

---

## 1. Scientific Scope

`CoChem-GEOM` bridges experimental CP-FTMW rotational spectroscopy (Pickett SPFIT/SPCAT `.var`/`.par`/`.fit`/`.lin` and `CoChem-SPYFY` JSON payloads) with converged canonical *ab initio* CCSD equilibrium structures ($r_e$) to produce transparent, provenance-verified structural comparisons for `CHEM 490` directed research and `CHEM 411` Physical Chemistry II:

- **Pickett SPFIT Ingestion (`cochem_geom.pickett`):**
  - Parses `.var` for true fitted rotational constants, fitted standard uncertainties, and reconstructs the exact parameter correlation matrix $R = L^\top L$ (and covariance matrix) from SPFIT's triangular Cholesky block.
  - Explicitly distinguishes `.par` a priori step limits from `.var` fitted uncertainties.
  - Supports Watson `A` and `S` reductions, `Ir`/`IIr`/`IIIr` axis representations, exact SPFIT parameter IDs (`10000`, `20000`, `30000`, `200`, `1100`, `2000`, `40100`, `41000`, `50000`, and hyperfine IDs), and quartic sign conventions (`-DJ`, `-DJK`, `-DK`).
  - Strictly rejects missing or zero uncertainties (`ValueError`; never silently substitutes `0.0`).
- **Weighted $r_0$ & Honest Kraitchman $r_s$ Fitting (`cochem_geom.r0_fitter`, `cochem_geom.kraitchman`):**
  - Single canonical CODATA rotational conversion constant `505379.008435 MHz·u·Å²`.
  - Passes experimental weights $w_i = 1/\sigma_i^2$ directly into `scipy.optimize.least_squares` residuals AND $(J^\top W J)^{-1}$ covariance estimation.
  - Preserves negative Kraitchman $r_s^2$ as explicit `imaginary_axis=True` Costain flags (never silently clips to `0.0`) and computes Costain systematic uncertainties $\sigma(|r_s|) = K / |r_s|$.
  - Applies exact rotational $g$-tensor electronic correction $I_\alpha(1 - (m_e/m_p) g_{\alpha\alpha})$ only when an explicit $g$-tensor is supplied; otherwise leaves electronic correction unset (`None`).
- **Converged Canonical CCSD Geometry Optimization (`cochem_geom.ccsd_driver`):**
  - **Student Public-Fork Parity (`pyscf_ccsd`):** CPU `PySCF` `ccsd.CCSD` analytic nuclear gradient + `geomeTRIC` + CCSD harmonic frequency verification.
  - **Workstation / CHEM 490 Research Parity (`orca_ccsd`):** `ORCA 6.1.1` canonical `! RHF CCSD cc-pVTZ TightOpt NumGrad NumFreq` (explicitly rejecting `DLPNO-CCSD(T)` geometry optimization).
  - Strict convergence verification requiring explicit convergence markers, $\max |\nabla E| \le 1.0 \times 10^{-5}\,E_h/\text{Bohr}$, `0` imaginary vibrational modes, and raw output SHA-256 provenance.
- **Separated Structural Comparison (`cochem_geom.reporting`):**
  - Emits JSON and LaTeX tables placing $r_e$ (CCSD/cc-pVTZ ORCA), $r_e$ (CCSD/cc-pVDZ PySCF), $r_s$ (Kraitchman), and $r_0$ (weighted fit) in distinct labeled columns, and marks $r_e^{\text{SE}}$ explicitly `PENDING` unless isotope-specific VPT2 $\Delta B_{\text{vib}}$ corrections are provided.

---

## 2. Installation & Verification

```bash
pip install --no-build-isolation -e .
python -m pytest tests/ -q
```

### CLI Usage (`cochem-geom`)

```bash
# Parse a Pickett .var / .par / .fit / .lin or CoChem-SPYFY .json deck
cochem-geom parse-pickett path/to/fa_dme_parent_spfit.var

# Reproduce T-GEOM-01 H2O structural comparison from verified on-disk CCSD outputs
cochem-geom benchmark-h2o \
  --exp-json benchmarks/h2o_t_geom_01/h2o_experimental_b0_piccardo2015.json \
  --out-dir benchmarks/h2o_t_geom_01 \
  --orca-out benchmarks/h2o_t_geom_01/orca_ccsd_cc_pvtz/h2o_ccsd_pvtz.out \
  --pyscf-out benchmarks/h2o_t_geom_01/pyscf_ccsd_cc_pvdz/h2o_ccsd_pvdz.log

# Parse real HCOOH...DME .var decks into the Rung-2 continuation bundle
cochem-geom bundle-hcooh-dme \
  --decks-dir path/to/pickett_decks \
  --out-dir benchmarks/hcooh_dme_rung2
```

### `CoChem-BASE` Provider Entry Point

`pyproject.toml` exposes `[project.entry-points."cochem.modules"]` `geom = "cochem_geom.provider:geom_provider"`.

---

## 3. Documentation & User Guide

- **[User Guide (`docs/USER_GUIDE.md`)](docs/USER_GUIDE.md)** — Worked CLI and Python examples for Pickett `.var`/`.par`/`.fit`/`.lin` and `CoChem-SPYFY` JSON ingestion, honest Kraitchman $r_s$ substitution coordinates, weighted $r_0$ fitting, canonical CCSD structural comparison, and Stage 2.1 `"C1"` symmetry-fallback warnings.

---

## 4. Benchmarks & Qualification Receipts

- `benchmarks/h2o_t_geom_01/`: Qualified `T-GEOM-01` `H2O` isotopologue benchmark (`Piccardo et al. JPCA 2015` SI Table 1 experimental $B_0$ + converged ORCA 6.1.1 `CCSD/cc-pVTZ` + converged PySCF CPU `CCSD/cc-pVDZ` + separated JSON/LaTeX structural comparison).
- `benchmarks/hcooh_dme_rung2/`: Rung-2 `HCOOH···DME` 5-isotopologue `.var` parsed bundle and Pipeline 5.0.0 continuation manifest (`CCSD` equilibrium geometry and $r_e^{\text{SE}}$ explicitly `PENDING`).
