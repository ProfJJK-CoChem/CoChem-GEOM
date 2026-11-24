# CoChem-GEOM User Guide

This guide walks through the core workflows in `CoChem-GEOM`:

1. [Pickett (`.var`/`.par`/`.fit`/`.lin`) & `CoChem-SPYFY` JSON Ingestion](#1-pickett--cochem-spyfy-json-ingestion)
2. [Honest Kraitchman Substitution ($r_s$) Coordinates](#2-honest-kraitchman-substitution-r_s-coordinates)
3. [Weighted Multi-Isotopologue $r_0$ Structural Fitting](#3-weighted-multi-isotopologue-r_0-structural-fitting)
4. [Converged Canonical CCSD & Separated Structural Comparison](#4-converged-canonical-ccsd--separated-structural-comparison)
5. [Stage 2.1 Symmetry Detection & Unverified `"C1"` Fallback Warnings](#5-stage-21-symmetry-detection--unverified-c1-fallback-warnings)

---

## 1. Pickett & `CoChem-SPYFY` JSON Ingestion

`cochem_geom.pickett` ingests JPL `calpgm` SPFIT/SPCAT files (`.var`, `.par`, `.fit`, `.lin`) as well as `CoChem-SPYFY` JSON spectroscopic constant payloads while enforcing strict uncertainty and sign-convention rules:

- **`.var` vs `.par` separation:** `.var` records carry `is_fitted_uncertainties=True` and reconstruct the exact parameter correlation matrix $R = L^\top L$ (and covariance matrix) from SPFIT's lower-triangular Cholesky block. `.par` records carry `is_fitted_uncertainties=False` (`uncertainty_source="par_prior_step_limit"`) so a priori step limits are never mistaken for fitted standard errors.
- **Strict uncertainty enforcement:** Missing, `None`, non-finite, or `<= 0` uncertainties raise `ValueError` immediately (no silent `0.0` defaults).
- **Hamiltonian reductions & representations:** Supports Watson `A` and `S` reductions and `Ir`/`IIr`/`IIIr` axis representations, mapping SPFIT quartic distortion terms (`-DJ`, `-DJK`, `-DK`) to positive physical values while retaining `raw_spfit_value_mhz`.

### CLI Example

```bash
# Parse a Pickett .var, .par, .fit, .lin, or CoChem-SPYFY .json file to JSON
cochem-geom parse-pickett path/to/fa_dme_parent_spfit.var
cochem-geom parse-pickett path/to/spyfy_constants.json --out parsed_deck.json
```

### Python Example

```python
from pathlib import Path
from cochem_geom.pickett import (
    parse_pickett_fit,
    parse_pickett_lin,
    parse_pickett_par,
    parse_pickett_var,
    parse_spyfy_json,
)

# 1. Pickett .var (fitted uncertainties + reconstructed correlation/covariance matrices)
deck = parse_pickett_var(Path("fa_dme_parent_spfit.var"))
print(deck.reduction, deck.representation, deck.is_fitted_uncertainties)
print(deck.rotational_constants())  # {'A': (val, unc), 'B': (val, unc), 'C': (val, unc)}

# 2. Pickett .par (prior step limits explicitly tagged is_fitted_uncertainties=False)
prior_deck = parse_pickett_par(Path("fa_dme_parent_spfit.par"))
assert prior_deck.is_fitted_uncertainties is False

# 3. Assigned transitions (.lin) and SPFIT output summary (.fit)
lines = parse_pickett_lin(Path("fa_dme_parent_spfit.lin"))
fit_summary = parse_pickett_fit(Path("fa_dme_parent_spfit.fit"))
print(f"{len(lines)} lines, MW RMS = {fit_summary.microwave_rms_mhz} MHz")

# 4. CoChem-SPYFY JSON payload
spyfy_deck = parse_spyfy_json(Path("spyfy_h2o_constants.json"))
```

Expected `CoChem-SPYFY` JSON schema:

```json
{
  "species_id": "H2O",
  "reduction": "A",
  "representation": "Ir",
  "optimized_constants": {
    "A": { "value": 835839.1, "uncertainty": 0.13 },
    "B": { "value": 435347.353, "uncertainty": 0.027 },
    "C": { "value": 278138.704, "uncertainty": 0.027 }
  }
}
```

---

## 2. Honest Kraitchman Substitution ($r_s$) Coordinates

`cochem_geom.kraitchman` solves Kraitchman's asymmetric-top equations using the single CODATA conversion factor `505379.008435 MHz·u·Å²`:

- **Never clips negative $r_s^2$:** When zero-point vibrational/inertial-defect contamination yields $r_s^2 < 0$ (e.g., the out-of-plane $c$-axis of planar $\text{H}_2\text{O}$), `imaginary_axis=True`, `abs_coord_angstrom=None`, `imaginary_magnitude_angstrom = sqrt(|r_s^2|)`, and an explicit Costain warning is emitted.
- **Costain systematic uncertainty:** Computes $\sigma_{\text{Costain}}(|r_s|) = K / |r_s|$ ($K = 0.0015\,\text{Å}^2$) alongside propagated experimental uncertainty.

### Python Example

```python
from cochem_geom.constants import get_exact_isotope_mass
from cochem_geom.kraitchman import (
    reconstruct_c2v_triatomic_rs,
    solve_kraitchman_from_rotational_constants,
)

# Single-atom substitution coordinate from experimental (A, B, C) in MHz
atom_o = solve_kraitchman_from_rotational_constants(
    parent_abc_mhz=(835839.1, 435347.353, 278138.704),
    iso_abc_mhz=(825366.6, 435353.5, 276948.8),
    parent_mass_u=get_exact_isotope_mass("16O") + 2 * get_exact_isotope_mass("1H"),
    delta_mass_u=get_exact_isotope_mass("18O") - get_exact_isotope_mass("16O"),
    atom_label="O",
    parent_unc_mhz=(0.13, 0.027, 0.027),
    iso_unc_mhz=(0.15, 0.03, 0.03),
)
print("O |b| (Å):", atom_o.axes["b"].abs_coord_angstrom)
print("O c-axis imaginary?:", atom_o.axes["c"].imaginary_axis, atom_o.axes["c"].warning)
```

For the qualified `T-GEOM-01` $\text{H}_2\text{O}$ benchmark (`benchmarks/h2o_t_geom_01/`), `reconstruct_c2v_triatomic_rs` yields:
- $r_s(\text{O–H}) = 0.96559 \pm 0.01436\,\text{Å}$
- $\angle_s(\text{H–O–H}) = 103.450 \pm 2.152^\circ$
- Both O and H out-of-plane $c$-axes explicitly flagged `imaginary_axis=True`.

---

## 3. Weighted Multi-Isotopologue $r_0$ Structural Fitting

`cochem_geom.r0_fitter` passes experimental weights $w_i = 1/\sigma_i^2$ directly into `scipy.optimize.least_squares` residuals ($\sqrt{w_i}(B_i^{\text{calc}} - B_i^{\text{obs}})$) and computes both unscaled $(J^\top W J)^{-1}$ and reduced-$\chi^2$-scaled covariance matrices with SVD condition-number diagnostics.

### Python Example

```python
import json
from pathlib import Path
from cochem_geom.r0_fitter import IsotopologueObservation, fit_c2v_triatomic_r0

raw = json.loads(
    Path("benchmarks/h2o_t_geom_01/h2o_experimental_b0_piccardo2015.json").read_text(encoding="utf-8")
)
isotopologues = [IsotopologueObservation.model_validate(item) for item in raw["isotopologues"]]

r0_res = fit_c2v_triatomic_r0(isotopologues)
print("r0(O-H):", r0_res.fitted_values["r(O-H)"], "±", r0_res.scaled_standard_errors["r(O-H)"], "Å")
print("angle(H-O-H):", r0_res.fitted_values["angle(H-O-H)"], "±", r0_res.scaled_standard_errors["angle(H-O-H)"], "deg")
print("RMS residual (MHz):", r0_res.rms_residual_mhz, "cond:", r0_res.condition_number)
```

For custom geometries beyond $C_{2v}$ triatomics, pass your own `geometry_builder(theta) -> np.ndarray` to `fit_weighted_structure(...)`. Semi-experimental $r_e^{\text{SE}}$ fits (`structure_type="re_se"`) are strictly gated: every isotopologue must supply verified VPT2 `vibrational_corrections_mhz` ($\Delta B_{\text{vib}}$), otherwise `ValueError` is raised and $r_e^{\text{SE}}$ remains explicitly `PENDING`.

---

## 4. Converged Canonical CCSD & Separated Structural Comparison

`cochem_geom.ccsd_driver` and `cochem_geom.reporting` compare canonical converged *ab initio* equilibrium structures ($r_e$) against experimental $r_s$ and $r_0$ without ever conflating $B_e$ and $B_0$:

- **ORCA backend (`orca_ccsd`):** Canonical `! RHF CCSD cc-pVTZ TightOpt NumGrad NumFreq` (explicitly forbids `DLPNO-CCSD(T)` geometry optimization).
- **PySCF backend (`pyscf_ccsd`):** CPU `PySCF` `ccsd.CCSD` analytic gradients + `geomeTRIC` + harmonic frequency verification (`n_imaginary_modes == 0`).
- **Convergence gates:** Explicit convergence markers, $\max |\nabla E| \le 1.0 \times 10^{-5}\,E_h/\text{Bohr}$, zero imaginary modes, and SHA-256 provenance of raw logs and `.xyz` files.

### CLI Example (`T-GEOM-01` $\text{H}_2\text{O}$ Benchmark)

```bash
cochem-geom benchmark-h2o \
  --exp-json benchmarks/h2o_t_geom_01/h2o_experimental_b0_piccardo2015.json \
  --out-dir benchmarks/h2o_t_geom_01 \
  --orca-out benchmarks/h2o_t_geom_01/orca_ccsd_cc_pvtz/h2o_ccsd_pvtz.out \
  --pyscf-out benchmarks/h2o_t_geom_01/pyscf_ccsd_cc_pvdz/h2o_ccsd_pvdz.log
```

Verified `T-GEOM-01` output columns written to `h2o_t_geom_01_comparison.json` and `h2o_t_geom_01_comparison.tex`:

| Parameter | $r_e$ (ORCA CCSD/cc-pVTZ) | $r_e$ (PySCF CCSD/cc-pVDZ) | $r_s$ (Kraitchman) | $r_0$ (weighted fit) | $r_e^{\text{SE}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $r(\text{O–H})$ (Å) | `0.95712` | `0.96435` | `0.96559 ± 0.01436` | `0.95747 ± 0.00119` | `PENDING` |
| $\angle(\text{H–O–H})$ ($^\circ$) | `103.893` | `102.210` | `103.450 ± 2.152` | `104.984 ± 0.164` | `PENDING` |

To bundle multi-isotopologue `.var`/`.lin`/`.fit` decks for downstream Pipeline 5.0.0 continuation (e.g., $\text{HCOOH}\cdots\text{DME}$):

```bash
cochem-geom bundle-hcooh-dme \
  --decks-dir path/to/pickett_decks \
  --out-dir benchmarks/hcooh_dme_rung2
```

---

## 5. Stage 2.1 Symmetry Detection & Unverified `"C1"` Fallback Warnings

`02-EVAL-AA/cochem_geom_eval_sym.py` (`SymmetryControllerUI`) detects molecular point groups via `molsym` and renders the Stage 2.1 Human-in-the-Loop (HITL) override dashboard alongside planar inertial defect checks ($\Delta I = I_c - I_a - I_b$).

**Mandatory User Notification on `"C1"` Fallback:**
Whenever `molsym` is unavailable or fails during point-group detection and `analyze_symmetry(...)` falls back to `"C1"`, `SymmetryControllerUI` **never fails silently**:

1. Emits a Python `UserWarning` via `warnings.warn(..., UserWarning)` stating that `"C1"` is an **unverified symmetry fallback** requiring manual verification/override.
2. Logs the event to the `CoChem_GEOM_Symmetry` logger.
3. Records the warning text on `sym_ui.last_symmetry_warning` and `sym_ui.symmetry_warnings[species_id]`.
4. Displays an explicit amber **`[Symmetry Fallback Warning — Unverified C1]`** banner in `render_hitl_dashboard(...)` and a per-species **`[Unverified C1 Fallback]`** badge next to the point-group dropdown.

### Python Example

```python
import warnings
import numpy as np
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

spec = spec_from_file_location("cochem_geom_eval_sym", Path("02-EVAL-AA/cochem_geom_eval_sym.py"))
eval_sym = module_from_spec(spec)
spec.loader.exec_module(eval_sym)

sym_ui = eval_sym.SymmetryControllerUI()
h2o_coords = np.array([
    [0.0, 0.0, 0.117790],
    [0.0, 0.755450, -0.471161],
    [0.0, -0.755450, -0.471161],
])
pg = sym_ui.analyze_symmetry(h2o_coords, ["O", "H", "H"], "H2O")
print("Detected point group:", pg, "| Warning:", sym_ui.last_symmetry_warning)

# If molsym is unavailable or encounters invalid input, UserWarning is always emitted:
sym_ui_no_molsym = eval_sym.SymmetryControllerUI(use_molsym=False)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fallback_pg = sym_ui_no_molsym.analyze_symmetry(h2o_coords, ["O", "H", "H"], "H2O")
print("Fallback PG:", fallback_pg, "| UserWarning count:", len(caught))
print("Recorded warning:", sym_ui_no_molsym.last_symmetry_warning)
```
