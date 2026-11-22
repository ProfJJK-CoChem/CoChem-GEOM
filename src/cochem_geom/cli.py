"""Command-line interface (`cochem-geom`) for CoChem-GEOM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .ccsd_driver import (
    parse_orca_ccsd_output,
    parse_pyscf_ccsd_output,
    run_orca_ccsd_optimization,
    run_pyscf_ccsd_optimization,
)
from .constants import get_exact_isotope_mass
from .kraitchman import reconstruct_c2v_triatomic_rs
from .pickett import (
    parse_pickett_fit,
    parse_pickett_lin,
    parse_pickett_par,
    parse_pickett_var,
    parse_spycfit_json,
)
from .r0_fitter import IsotopologueObservation, fit_c2v_triatomic_r0
from .reporting import (
    build_structural_comparison_payload,
    write_structural_comparison_artifacts,
)


def _load_h2o_isotopologues(exp_json_path: Path) -> tuple[list[IsotopologueObservation], dict[str, Any]]:
    raw = json.loads(exp_json_path.read_text(encoding="utf-8"))
    isos = [IsotopologueObservation.model_validate(item) for item in raw["isotopologues"]]
    return isos, raw.get("provenance", {})


def _cmd_parse_pickett(args: argparse.Namespace) -> int:
    path = Path(args.path)
    suffix = path.suffix.lower()
    if suffix == ".var":
        out = parse_pickett_var(path).model_dump(mode="json")
    elif suffix == ".par":
        out = parse_pickett_par(path).model_dump(mode="json")
    elif suffix == ".fit":
        out = parse_pickett_fit(path).model_dump(mode="json")
    elif suffix == ".lin":
        out = {"lines": [ln.model_dump(mode="json") for ln in parse_pickett_lin(path)]}
    elif suffix == ".json":
        out = parse_spycfit_json(path).model_dump(mode="json")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    rendered = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


def _cmd_benchmark_h2o(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_path = Path(args.exp_json).resolve()
    isotopologues, exp_prov = _load_h2o_isotopologues(exp_path)

    # 1. Weighted r_0 fit across H2O, H2(18O), D2O, HDO
    r0_res = fit_c2v_triatomic_r0(isotopologues)

    # 2. Honest Kraitchman r_s from parent H2O + H2(18O) + HDO
    iso_map = {iso.species_id: iso for iso in isotopologues}
    parent = iso_map["H2O"]
    o18 = iso_map["H2(18O)"]
    hdo = iso_map["HDO"]
    rs_res = reconstruct_c2v_triatomic_rs(
        parent_abc_mhz=parent.rotational_constants_mhz,
        parent_unc_mhz=parent.uncertainties_mhz,
        central_sub_abc_mhz=o18.rotational_constants_mhz,
        central_sub_unc_mhz=o18.uncertainties_mhz,
        terminal_sub_abc_mhz=hdo.rotational_constants_mhz,
        terminal_sub_unc_mhz=hdo.uncertainties_mhz,
        mass_central_u=get_exact_isotope_mass("16O"),
        mass_central_iso_u=get_exact_isotope_mass("18O"),
        mass_terminal_u=get_exact_isotope_mass("1H"),
        mass_terminal_iso_u=get_exact_isotope_mass("2H"),
        central_label="O",
        terminal_label="H",
    )

    # 3. Converged CCSD geometries (either run live or parse verified on-disk logs)
    h2o_init = np.array(
        [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ],
        dtype=np.float64,
    )

    orca_res = None
    if args.orca_out:
        orca_p = Path(args.orca_out).resolve()
        orca_xyz = orca_p.with_suffix(".xyz")
        orca_res = parse_orca_ccsd_output(
            orca_p, xyz_path=orca_xyz if orca_xyz.is_file() else None
        )
    elif args.run_qm:
        orca_res = run_orca_ccsd_optimization(
            work_dir=out_dir / "orca_ccsd_cc_pvtz",
            job_name="h2o_ccsd_pvtz",
            symbols=["O", "H", "H"],
            initial_coords_ang=h2o_init,
            basis="cc-pVTZ",
        )

    pyscf_res = None
    if args.pyscf_out:
        pyscf_p = Path(args.pyscf_out).resolve()
        pyscf_xyz = pyscf_p.with_suffix(".xyz")
        pyscf_res = parse_pyscf_ccsd_output(
            pyscf_p, xyz_path=pyscf_xyz if pyscf_xyz.is_file() else None
        )
    elif args.run_qm:
        pyscf_res = run_pyscf_ccsd_optimization(
            work_dir=out_dir / "pyscf_ccsd_cc_pvdz",
            job_name="h2o_ccsd_pvdz",
            symbols=["O", "H", "H"],
            initial_coords_ang=h2o_init,
            basis="cc-pVDZ",
        )

    payload = build_structural_comparison_payload(
        molecule_name="H2O",
        symmetry="C2v",
        r0_result=r0_res,
        rs_result=rs_res,
        orca_ccsd=orca_res,
        pyscf_ccsd=pyscf_res,
        re_se_result=None,
        experimental_provenance=exp_prov,
    )
    json_path, tex_path = write_structural_comparison_artifacts(out_dir, "h2o_t_geom_01", payload)
    print(json.dumps({"json_report": str(json_path), "latex_report": str(tex_path)}, indent=2))
    return 0


def _cmd_bundle_hcooh_dme(args: argparse.Namespace) -> int:
    decks_dir = Path(args.decks_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    deck_names = [
        "fa_dme_parent_spfit",
        "h13cooh_dme",
        "hcood_dme",
        "dcooh_dme_2h_hfs",
        "fa_13ch3och3",
    ]
    parsed_decks: dict[str, Any] = {}
    for stem in deck_names:
        var_path = decks_dir / f"{stem}.var"
        lin_path = decks_dir / f"{stem}.lin"
        fit_path = decks_dir / f"{stem}.fit"
        deck = parse_pickett_var(var_path)
        entry: dict[str, Any] = {
            "var": deck.model_dump(mode="json"),
        }
        if lin_path.is_file():
            lines = parse_pickett_lin(lin_path)
            entry["n_assigned_lines"] = len(lines)
        if fit_path.is_file():
            fit_summary = parse_pickett_fit(fit_path)
            entry["fit_summary"] = {
                "microwave_rms_mhz": fit_summary.microwave_rms_mhz,
                "microwave_avg_mhz": fit_summary.microwave_avg_mhz,
                "rotor_type": fit_summary.rotor_type,
                "n_transitions_parsed": len(fit_summary.transitions),
                "source_sha256": fit_summary.source_sha256,
            }
        parsed_decks[stem] = entry

    bundle_path = out_dir / "hcooh_dme_parsed_decks.json"
    bundle_payload = {
        "schema_version": "cochem.geom.rung2-pickett-bundle/1",
        "complex": "HCOOH...DME",
        "decks_directory": str(decks_dir),
        "n_isotopologues": len(parsed_decks),
        "isotopologues": parsed_decks,
        "ccsd_equilibrium_geometry_status": "PENDING",
        "re_se_status": "PENDING",
    }
    bundle_path.write_text(json.dumps(bundle_payload, indent=2) + "\n", encoding="utf-8")

    manifest_path = out_dir / "pipeline_500_continuation_manifest.json"
    manifest_payload = {
        "schema_version": "cochem.geom.pipeline500-continuation/1",
        "target_system": "HCOOH...DME (14 atoms, 5 FTMW isotopologues)",
        "current_status": {
            "ftmw_pickett_var_ingestion": "VERIFIED (5/5 real .var decks parsed with fitted uncertainties & correlations)",
            "ccsd_equilibrium_geometry": "PENDING",
            "vpt2_anharmonic_delta_b_vib": "PENDING",
            "re_se_structural_fit": "PENDING",
        },
        "blocking_reason": (
            "Zero converged CCSD/CCSD(T) geometry optimizations exist on disk for HCOOH...DME "
            "(legacy ORCA DLPNO-CCSD(T) Opt jobs failed because ORCA MDCI lacks analytic DLPNO gradients). "
            "Must be submitted through Pipeline 5.0.0 cloud/workstation runner."
        ),
        "recommended_pipeline_500_jobs": [
            {
                "job_id": "hcooh_dme_ccsd_cc_pvdz_opt_freq",
                "backend": "orca_ccsd_or_pyscf_gpu",
                "method": "canonical CCSD/cc-pVDZ TightOpt Freq (or CCSD(T) numerical gradient on HPC)",
                "convergence_gates": {
                    "max_gradient_eh_bohr_le": 1.0e-5,
                    "n_imaginary_modes": 0,
                },
            }
        ],
        "parsed_bundle_file": bundle_path.name,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "parsed_bundle": str(bundle_path),
                "continuation_manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cochem-geom",
        description="CoChem-GEOM FTMW rotational spectroscopy fitting and CCSD geometry comparison CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse-pickett", help="Parse a Pickett .var/.par/.fit/.lin or SpycFit .json file.")
    p_parse.add_argument("path", help="Path to input spectroscopic file.")
    p_parse.add_argument("--out", help="Optional path to write JSON output.")
    p_parse.set_defaults(func=_cmd_parse_pickett)

    p_h2o = sub.add_parser("benchmark-h2o", help="Run or assemble the T-GEOM-01 H2O structural comparison.")
    p_h2o.add_argument("--exp-json", required=True, help="Path to H2O experimental B0 JSON packet.")
    p_h2o.add_argument("--out-dir", required=True, help="Output directory for JSON and LaTeX reports.")
    p_h2o.add_argument("--orca-out", help="Path to existing converged ORCA CCSD .out file.")
    p_h2o.add_argument("--pyscf-out", help="Path to existing converged PySCF CCSD .log file.")
    p_h2o.add_argument("--run-qm", action="store_true", help="Execute live ORCA + PySCF CCSD optimizations.")
    p_h2o.set_defaults(func=_cmd_benchmark_h2o)

    p_dme = sub.add_parser("bundle-hcooh-dme", help="Parse the 5 HCOOH...DME .var decks into a Rung-2 bundle.")
    p_dme.add_argument("--decks-dir", required=True, help="Directory containing real Pickett decks.")
    p_dme.add_argument("--out-dir", required=True, help="Output directory for Rung-2 bundle and manifest.")
    p_dme.set_defaults(func=_cmd_bundle_hcooh_dme)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
