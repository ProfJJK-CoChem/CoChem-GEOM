"""Tests for converged CCSD outputs, separated JSON/LaTeX reporting, provider & repo hygiene."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from cochem_geom.ccsd_driver import (
    generate_orca_ccsd_input,
    parse_orca_ccsd_output,
    parse_pyscf_ccsd_output,
)
from cochem_geom.cli import main as cli_main
from cochem_geom.provider import geom_provider

REPO_ROOT = Path(__file__).resolve().parent.parent
H2O_BENCH_DIR = REPO_ROOT / "benchmarks" / "h2o_t_geom_01"
DME_BENCH_DIR = REPO_ROOT / "benchmarks" / "hcooh_dme_rung2"


def test_orca_dlpno_opt_is_explicitly_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="DLPNO-CCSD"):
        generate_orca_ccsd_input(
            tmp_path / "bad.inp",
            ["O", "H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.75, -0.5], [0.0, -0.75, -0.5]]),
            method="DLPNO-CCSD(T)",
        )


def test_real_converged_h2o_orca_ccsd_cc_pvtz() -> None:
    orca_out = H2O_BENCH_DIR / "orca_ccsd_cc_pvtz" / "h2o_ccsd_pvtz.out"
    orca_xyz = H2O_BENCH_DIR / "orca_ccsd_cc_pvtz" / "h2o_ccsd_pvtz.xyz"
    assert orca_out.is_file(), f"Missing real ORCA CCSD/cc-pVTZ output at {orca_out}"

    res = parse_orca_ccsd_output(orca_out, xyz_path=orca_xyz if orca_xyz.is_file() else None)
    assert res.backend == "orca_ccsd"
    assert res.converged_marker_found is True
    assert res.normal_termination_found is True
    assert res.max_gradient_eh_bohr <= 1.0e-5
    assert res.n_imaginary_modes == 0
    assert len(res.vibrational_frequencies_cm1) == 3
    assert all(f > 1000.0 for f in res.vibrational_frequencies_cm1)
    assert len(res.output_sha256) == 64

    # NIST CCCBDB CCSD/cc-pVTZ H2O: r_e(O-H) = 0.9589 A, angle_e(HOH) = 104.2 deg
    r_oh = res.internal_geometry["r(O-H)_mean"]
    ang = res.internal_geometry["angle(H-O-H)"]
    assert abs(r_oh - 0.9589) <= 0.005
    assert abs(ang - 104.2) <= 0.5


def test_real_converged_h2o_pyscf_ccsd_cc_pvdz() -> None:
    pyscf_log = H2O_BENCH_DIR / "pyscf_ccsd_cc_pvdz" / "h2o_ccsd_pvdz.log"
    pyscf_xyz = H2O_BENCH_DIR / "pyscf_ccsd_cc_pvdz" / "h2o_ccsd_pvdz.xyz"
    assert pyscf_log.is_file(), f"Missing real PySCF CCSD/cc-pVDZ log at {pyscf_log}"

    res = parse_pyscf_ccsd_output(pyscf_log, xyz_path=pyscf_xyz if pyscf_xyz.is_file() else None)
    assert res.backend == "pyscf_ccsd"
    assert res.converged_marker_found is True
    assert res.normal_termination_found is True
    assert res.max_gradient_eh_bohr <= 1.0e-5
    assert res.n_imaginary_modes == 0
    assert len(res.vibrational_frequencies_cm1) == 3
    assert all(f > 1000.0 for f in res.vibrational_frequencies_cm1)
    assert len(res.output_sha256) == 64

    # NIST CCCBDB CCSD/cc-pVDZ H2O: r_e(O-H) = 0.9649 A, angle_e(HOH) = 102.0 deg
    r_oh = res.internal_geometry["r(O-H)_mean"]
    ang = res.internal_geometry["angle(H-O-H)"]
    assert abs(r_oh - 0.9649) <= 0.005
    assert abs(ang - 102.0) <= 0.5


def test_cli_separated_json_and_latex_comparison(tmp_path: Path) -> None:
    rc = cli_main(
        [
            "benchmark-h2o",
            "--exp-json",
            str(H2O_BENCH_DIR / "h2o_experimental_b0_piccardo2015.json"),
            "--out-dir",
            str(tmp_path),
            "--orca-out",
            str(H2O_BENCH_DIR / "orca_ccsd_cc_pvtz" / "h2o_ccsd_pvtz.out"),
            "--pyscf-out",
            str(H2O_BENCH_DIR / "pyscf_ccsd_cc_pvdz" / "h2o_ccsd_pvdz.log"),
        ]
    )
    assert rc == 0

    json_file = tmp_path / "h2o_t_geom_01_comparison.json"
    tex_file = tmp_path / "h2o_t_geom_01_comparison.tex"
    assert json_file.is_file()
    assert tex_file.is_file()

    payload = json.loads(json_file.read_text(encoding="utf-8"))
    cols = payload["columns"]
    assert cols["r_e_ccsd_cc_pvtz_orca"]["status"] == "VERIFIED"
    assert cols["r_e_ccsd_cc_pvdz_pyscf"]["status"] == "VERIFIED"
    assert cols["r_s_kraitchman"]["status"] == "VERIFIED"
    assert cols["r_0_weighted_fit"]["status"] == "VERIFIED"
    assert cols["r_e_se"]["status"] == "PENDING"
    assert payload["electronic_g_tensor_correction"] is None

    tex = tex_file.read_text(encoding="utf-8")
    assert "$r_e$ (CCSD/cc-pVTZ ORCA)" in tex
    assert "$r_e$ (CCSD/cc-pVDZ PySCF)" in tex
    assert "$r_s$ (Kraitchman)" in tex
    assert "$r_0$ (weighted fit)" in tex
    assert "PENDING" in tex
    assert "855881.3" not in tex
    assert "855881.3" not in json_file.read_text(encoding="utf-8")


def test_rung2_hcooh_dme_bundle_and_continuation_manifest() -> None:
    bundle_path = DME_BENCH_DIR / "hcooh_dme_parsed_decks.json"
    manifest_path = DME_BENCH_DIR / "pipeline_500_continuation_manifest.json"
    assert bundle_path.is_file()
    assert manifest_path.is_file()

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["n_isotopologues"] == 5
    assert bundle["ccsd_equilibrium_geometry_status"] == "PENDING"
    assert bundle["re_se_status"] == "PENDING"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["current_status"]["ccsd_equilibrium_geometry"] == "PENDING"


def test_cochem_base_geom_provider_entry_point() -> None:
    meta = geom_provider.metadata()
    assert meta["module_id"] == "geom"
    assert meta["distribution"] == "CoChem-GEOM"
    assert "parse_pickett" in meta["operations"]
    assert "fit_weighted_r0" in meta["operations"]
    assert "optimize_ccsd" in meta["operations"]
    assert geom_provider() == meta


def test_zero_ast_syntax_errors_and_zero_hallucinated_water_constant() -> None:
    py_files = [
        p
        for p in REPO_ROOT.rglob("*.py")
        if ".git" not in p.parts and ".mypy_cache" not in p.parts and "build" not in p.parts
    ]
    assert len(py_files) >= 15
    for py_file in py_files:
        source = py_file.read_text(encoding="utf-8")
        ast.parse(source, filename=str(py_file))

    # Reject hallucinated Water rotational constant 855881.3 anywhere in tracked code/docs/benchmarks
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".mypy_cache", ".pytest_cache", "__pycache__"} for part in path.parts):
            continue
        if path.name == "test_ccsd_and_reporting.py":
            continue
        if path.suffix.lower() in {".py", ".tex", ".json", ".md", ".txt", ".inp"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "855881.3" not in text, f"Hallucinated Water constant 855881.3 found in {path}"
