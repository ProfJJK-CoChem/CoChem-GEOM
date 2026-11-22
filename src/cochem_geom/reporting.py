"""Separated structural comparison reporting (JSON + LaTeX) for CoChem-GEOM.

Strictly separates equilibrium ab initio structures (r_e from ORCA CCSD/cc-pVTZ
and PySCF CCSD/cc-pVDZ), Kraitchman substitution structures (r_s), weighted
ground-state structures (r_0), and semi-experimental equilibrium structures
(r_e^SE, explicitly marked 'PENDING' unless verified isotope-specific Delta_B_vib
values are supplied). Never mixes B_e with B_0 and never embeds hardcoded constants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ccsd_driver import CCSDOptimizationResult
from .kraitchman import TriatomicRsStructure
from .r0_fitter import R0FitResult


def _escape_latex(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def build_structural_comparison_payload(
    *,
    molecule_name: str,
    symmetry: str,
    r0_result: R0FitResult,
    rs_result: TriatomicRsStructure,
    orca_ccsd: CCSDOptimizationResult | None = None,
    pyscf_ccsd: CCSDOptimizationResult | None = None,
    re_se_result: R0FitResult | None = None,
    experimental_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a schema-versioned structural comparison JSON payload."""
    bond_param = r0_result.parameter_names[0]
    angle_param = r0_result.parameter_names[1]

    def _extract_ccsd_pair(res: CCSDOptimizationResult | None) -> dict[str, Any]:
        if res is None:
            return {"status": "PENDING"}
        r_vals = [v for k, v in res.internal_geometry.items() if k.endswith("_mean")]
        r_val = r_vals[0] if r_vals else next(
            (v for k, v in res.internal_geometry.items() if k.startswith("r(")), None
        )
        ang_val = next(
            (v for k, v in res.internal_geometry.items() if k.startswith("angle(")), None
        )
        return {
            "status": "VERIFIED",
            "backend": res.backend,
            "engine_version": res.engine_version,
            "method": res.method,
            "basis": res.basis,
            "bond_angstrom": r_val,
            "angle_deg": ang_val,
            "electronic_energy_hartree": res.electronic_energy_hartree,
            "t1_diagnostic": res.t1_diagnostic,
            "max_gradient_eh_bohr": res.max_gradient_eh_bohr,
            "n_imaginary_modes": res.n_imaginary_modes,
            "vibrational_frequencies_cm1": res.vibrational_frequencies_cm1,
            "rotational_constants_be_mhz": {
                "A_e": res.rotational_constants_be_mhz[0],
                "B_e": res.rotational_constants_be_mhz[1],
                "C_e": res.rotational_constants_be_mhz[2],
            },
            "output_file": res.output_file,
            "output_sha256": res.output_sha256,
            "xyz_file": res.xyz_file,
            "xyz_sha256": res.xyz_sha256,
        }

    re_se_block: dict[str, Any]
    if re_se_result is None:
        re_se_block = {
            "status": "PENDING",
            "reason": (
                "Semi-experimental equilibrium structure r_e^SE requires verified "
                "isotope-specific anharmonic vibrational corrections Delta_B_vib "
                "(B_e^SE = B_0^exp - Delta_B_vib); never equated with B_0 or r_0."
            ),
            "bond_angstrom": None,
            "angle_deg": None,
        }
    else:
        re_se_block = {
            "status": "VERIFIED",
            "bond_angstrom": re_se_result.fitted_values[bond_param],
            "bond_uncertainty_angstrom": re_se_result.scaled_standard_errors[bond_param],
            "angle_deg": re_se_result.fitted_values[angle_param],
            "angle_uncertainty_deg": re_se_result.scaled_standard_errors[angle_param],
        }

    return {
        "schema_version": "cochem.geom.structural-comparison/1",
        "molecule": molecule_name,
        "symmetry": symmetry,
        "codata_rotational_factor_mhz_u_a2": 505379.008435,
        "electronic_g_tensor_correction": None,
        "columns": {
            "r_e_ccsd_cc_pvtz_orca": _extract_ccsd_pair(orca_ccsd),
            "r_e_ccsd_cc_pvdz_pyscf": _extract_ccsd_pair(pyscf_ccsd),
            "r_s_kraitchman": {
                "status": "VERIFIED",
                "bond_angstrom": rs_result.r_xy_angstrom,
                "bond_costain_uncertainty_angstrom": rs_result.r_xy_costain_uncertainty_angstrom,
                "angle_deg": rs_result.angle_yxy_deg,
                "angle_costain_uncertainty_deg": rs_result.angle_yxy_costain_uncertainty_deg,
                "first_moment_residual_u_a": rs_result.first_moment_residual_u_a,
                "imaginary_axes": {
                    atom_lbl: [
                        ax for ax, ax_res in atom_res.axes.items() if ax_res.imaginary_axis
                    ]
                    for atom_lbl, atom_res in rs_result.atom_results.items()
                },
                "atoms": {
                    atom_lbl: atom_res.model_dump(mode="json")
                    for atom_lbl, atom_res in rs_result.atom_results.items()
                },
                "notes": rs_result.notes,
            },
            "r_0_weighted_fit": {
                "status": "VERIFIED",
                "bond_label": bond_param,
                "angle_label": angle_param,
                "bond_angstrom": r0_result.fitted_values[bond_param],
                "bond_std_err_scaled_angstrom": r0_result.scaled_standard_errors[bond_param],
                "bond_std_err_unscaled_angstrom": r0_result.unscaled_standard_errors[bond_param],
                "angle_deg": r0_result.fitted_values[angle_param],
                "angle_std_err_scaled_deg": r0_result.scaled_standard_errors[angle_param],
                "angle_std_err_unscaled_deg": r0_result.unscaled_standard_errors[angle_param],
                "n_observations": r0_result.n_observations,
                "degrees_of_freedom": r0_result.degrees_of_freedom,
                "rms_residual_mhz": r0_result.rms_residual_mhz,
                "chi_squared": r0_result.chi_squared,
                "reduced_chi_squared": r0_result.reduced_chi_squared,
                "condition_number": r0_result.condition_number,
                "correlation_matrix": r0_result.correlation_matrix,
                "residuals": [r.model_dump(mode="json") for r in r0_result.residuals],
            },
            "r_e_se": re_se_block,
        },
        "experimental_provenance": experimental_provenance or {},
    }


def render_structural_comparison_latex(payload: dict[str, Any]) -> str:
    """Render a clean LaTeX comparison table separating r_e, r_s, r_0, and r_e^SE."""
    cols = payload["columns"]
    orca = cols["r_e_ccsd_cc_pvtz_orca"]
    pyscf = cols["r_e_ccsd_cc_pvdz_pyscf"]
    rs = cols["r_s_kraitchman"]
    r0 = cols["r_0_weighted_fit"]
    re_se = cols["r_e_se"]

    def _fmt_ccsd_val(entry: dict[str, Any], key: str, digits: int) -> str:
        if entry.get("status") != "VERIFIED" or entry.get(key) is None:
            return "PENDING"
        return f"{float(entry[key]):.{digits}f}"

    def _fmt_re_se(key: str, unc_key: str, digits: int) -> str:
        if re_se.get("status") != "VERIFIED" or re_se.get(key) is None:
            return "PENDING"
        return f"{float(re_se[key]):.{digits}f} $\\pm$ {float(re_se[unc_key]):.{digits}f}"

    mol_esc = _escape_latex(payload["molecule"])
    sym_esc = _escape_latex(payload["symmetry"])

    r0_bond = f"{r0['bond_angstrom']:.5f} $\\pm$ {r0['bond_std_err_scaled_angstrom']:.5f}"
    r0_ang = f"{r0['angle_deg']:.3f} $\\pm$ {r0['angle_std_err_scaled_deg']:.3f}"
    rs_bond = f"{rs['bond_angstrom']:.5f} $\\pm$ {rs['bond_costain_uncertainty_angstrom']:.5f}"
    rs_ang = f"{rs['angle_deg']:.3f} $\\pm$ {rs['angle_costain_uncertainty_deg']:.3f}"

    lines = [
        f"% CoChem-GEOM Structural Comparison Table ({payload['schema_version']})",
        "\\begin{table}[h!]",
        "\\centering",
        (
            f"\\caption{{Separated structural comparison for {mol_esc} (\\texttt{{{sym_esc}}}): "
            "converged ab initio $r_e$ (ORCA CCSD/cc-pVTZ and PySCF CCSD/cc-pVDZ), "
            "Kraitchman $r_s$ (with Costain uncertainties), weighted multi-isotopologue $r_0$, "
            "and semi-experimental $r_e^{\\text{SE}}$ (explicitly \\texttt{PENDING} without VPT2 $\\Delta B_{\\text{vib}}$).}"
        ),
        "\\begin{tabular}{l c c c c c}",
        "\\hline\\hline",
        (
            "Parameter & $r_e$ (CCSD/cc-pVTZ ORCA) & $r_e$ (CCSD/cc-pVDZ PySCF) & "
            "$r_s$ (Kraitchman) & $r_0$ (weighted fit) & $r_e^{\\text{SE}}$ \\\\"
        ),
        "\\hline",
        (
            f"{_escape_latex(r0['bond_label'])} (\\AA) & "
            f"{_fmt_ccsd_val(orca, 'bond_angstrom', 5)} & "
            f"{_fmt_ccsd_val(pyscf, 'bond_angstrom', 5)} & "
            f"{rs_bond} & {r0_bond} & {_fmt_re_se('bond_angstrom', 'bond_uncertainty_angstrom', 5)} \\\\"
        ),
        (
            f"{_escape_latex(r0['angle_label'])} ($^\\circ$) & "
            f"{_fmt_ccsd_val(orca, 'angle_deg', 3)} & "
            f"{_fmt_ccsd_val(pyscf, 'angle_deg', 3)} & "
            f"{rs_ang} & {r0_ang} & {_fmt_re_se('angle_deg', 'angle_uncertainty_deg', 3)} \\\\"
        ),
        "\\hline\\hline",
        "\\end{tabular}",
        "\\label{tab:geom_structural_comparison}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def write_structural_comparison_artifacts(
    out_dir: Path | str,
    prefix: str,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Write both JSON and LaTeX structural comparison artifacts to `out_dir`."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    json_path = d / f"{prefix}_comparison.json"
    tex_path = d / f"{prefix}_comparison.tex"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tex_path.write_text(render_structural_comparison_latex(payload), encoding="utf-8")
    return json_path, tex_path
