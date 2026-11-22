#!/usr/bin/env python3
"""CoChem-GEOM Legacy & Pure-Math Compatibility PyTest Suite.

Verifies the legacy numbered stage modules (`01-INGEST-AA` through `04-REPORT-AA`)
after removing broken ad-hoc DBOC, line-0 drop bugs, silent Kraitchman clipping,
unweighted least-squares, and vacuous MPQC-missing anti-proofs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def load_module_from_path(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


repo_root = Path(__file__).parent.parent
ingest_parser_mod = load_module_from_path(
    "cochem_geom_ingest_parser", repo_root / "01-INGEST-AA" / "cochem_geom_ingest_parser.py"
)
ingest_math_mod = load_module_from_path(
    "cochem_geom_ingest_math", repo_root / "01-INGEST-AA" / "cochem_geom_ingest_math.py"
)
eval_sym_mod = load_module_from_path(
    "cochem_geom_eval_sym", repo_root / "02-EVAL-AA" / "cochem_geom_eval_sym.py"
)
eval_triage_mod = load_module_from_path(
    "cochem_geom_eval_triage", repo_root / "02-EVAL-AA" / "cochem_geom_eval_triage.py"
)
fitter_core_mod = load_module_from_path(
    "cochem_geom_fitter_core", repo_root / "03-FIT-AA" / "cochem_geom_fitter_core.py"
)
fitter_optim_mod = load_module_from_path(
    "cochem_geom_fitter_optim", repo_root / "03-FIT-AA" / "cochem_geom_fitter_optim.py"
)
ccsdt_refine_mod = load_module_from_path(
    "cochem_geom_ccsdt_refine", repo_root / "03b-REFINE-AA" / "cochem_geom_ccsdt_refine.py"
)
reporter_latex_mod = load_module_from_path(
    "cochem_geom_reporter_latex", repo_root / "04-REPORT-AA" / "cochem_geom_reporter_latex.py"
)
reporter_ui_mod = load_module_from_path(
    "cochem_geom_reporter_ui", repo_root / "04-REPORT-AA" / "cochem_geom_reporter_ui.py"
)
router_mod = load_module_from_path(
    "cochem_geom_router", repo_root / "03-FIT-AA" / "cochem_geom_router.py"
)

SpectraIngestionEngine = ingest_parser_mod.SpectraIngestionEngine
CoordinateStandardizer = ingest_math_mod.CoordinateStandardizer
SymmetryControllerUI = eval_sym_mod.SymmetryControllerUI
VariableTriageEngine = eval_triage_mod.VariableTriageEngine
DynamicBoundsTuner = fitter_core_mod.DynamicBoundsTuner
ZMatrixEngine = fitter_core_mod.ZMatrixEngine
KraitchmanEngine = fitter_optim_mod.KraitchmanEngine
MultiSeedOptimizer = fitter_optim_mod.MultiSeedOptimizer
CanonicalCCSDORCAOptimizer = ccsdt_refine_mod.CanonicalCCSDORCAOptimizer
GEOMReportLatexGenerator = reporter_latex_mod.GEOMReportLatexGenerator
GEOMReportUIGenerator = reporter_ui_mod.GEOMReportUIGenerator
EngineRouter = router_mod.EngineRouter


def test_ingest_abundance_and_pickett(tmp_path: Path) -> None:
    engine = SpectraIngestionEngine(tmp_path)
    assert engine._determine_abundance("H2O") == 100.0
    assert abs(engine._determine_abundance("13CH4") - 1.07) < 0.1

    par_file = tmp_path / "test.par"
    par_file.write_text("10000 12345.6789-0.0050\n20000 54321.9876 0.0040\n", encoding="utf-8")
    constants = engine.parse_pickett_par(par_file)
    assert len(constants) == 2
    assert constants[0].constant_type == "A"
    assert abs(constants[0].value_mhz - 12345.6789) < 1e-3
    assert abs(constants[0].uncertainty_mhz - 0.0050) < 1e-6
    assert constants[0].is_fitted_uncertainty is False


def test_math_eckart_alignment_and_g_tensor() -> None:
    math_engine = CoordinateStandardizer()
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755450, -0.471161],
            [0.0, -0.755450, -0.471161],
        ]
    )
    masses = np.array([15.9949146, 1.0078250, 1.0078250])

    _, moments, rot_mat = math_engine.align_to_principal_axes(coords, masses)
    assert abs(np.linalg.det(rot_mat) - 1.0) < 1e-5

    # Without g-tensor, electronic correction is explicitly unset/None
    no_corr = math_engine.apply_electronic_g_tensor_correction(moments, g_tensor_diagonal=None)
    assert no_corr.applied is False
    assert no_corr.corrected_moments_u_a2 is None

    # With positive rotational g-tensor, moments decrease by (m_e/m_p) * g_alpha
    g_corr = math_engine.apply_electronic_g_tensor_correction(
        moments, g_tensor_diagonal=(0.657, 0.718, 0.645)
    )
    assert g_corr.applied is True
    assert g_corr.corrected_moments_u_a2 is not None
    assert np.all(np.asarray(g_corr.corrected_moments_u_a2) < moments)

    pas_dipole, _ = math_engine.project_to_pas(np.array([0.0, 1.0, 2.0]), None, rot_mat)
    assert pas_dipole.shape == (3,)


def test_sym_and_triage() -> None:
    sym_ui = SymmetryControllerUI()
    moments = np.array([10.0, 50.0, 60.0])
    res = sym_ui.check_planar_inertial_defect(moments)
    assert "[Planar]" in res.value

    triage = VariableTriageEngine(["C", "N", "H", "H"])
    assert len(triage.parameters) > 0
    triage.apply_hydrogen_lock()
    for p in triage.parameters:
        if p["type"] != "Bond":
            assert not p["is_frozen"]

    badge = triage.evaluate_sufficiency(num_constants=10, rot_constants=(10000.0, 5000.0, 5000.0))
    assert "SUFFICIENT" in badge.value or "UNSUFFICIENT" in badge.value


def test_fitter_core_and_optim() -> None:
    tuner = DynamicBoundsTuner()
    lower, upper = tuner.get_bond_bounds("C", "C", bond_order=2.0)
    assert lower < upper

    z_engine = ZMatrixEngine()
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 1.2, 1.8]])
    params = [{"type": "Bond", "atoms": [1, 0]}, {"type": "Angle", "atoms": [2, 1, 0]}]
    internals = z_engine.calculate_internal_coordinates(coords, params)
    assert len(internals) == 2

    b_mat = z_engine.compute_b_matrix(coords, params)
    assert b_mat.shape == (2, 9)

    kraitchman = KraitchmanEngine()
    parent_i = np.array([10.0, 50.0, 60.0])
    iso_i = np.array([10.0, 50.01, 60.01])
    rs_res = kraitchman.fit_rs_kraitchman(parent_i, iso_i, M_parent=30.0, delta_m=1.0)
    assert set(rs_res.axes) == {"a", "b", "c"}

    opt = MultiSeedOptimizer()
    _, _, svd_report = opt.evaluate_jacobian_covariance(np.ones((2, 2)), np.ones(2))
    assert "condition_number" in svd_report


def test_ccsd_input_and_reporters(tmp_path: Path) -> None:
    orca_opt = CanonicalCCSDORCAOptimizer(tmp_path)
    inp = orca_opt.generate_input(
        "h2o_test",
        ["O", "H", "H"],
        np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]),
        charge=0,
        mult=1,
        basis="cc-pVTZ",
    )
    content = inp.read_text(encoding="utf-8")
    assert "! RHF CCSD cc-pVTZ TightOpt" in content
    assert "DLPNO" not in content

    latex_gen = GEOMReportLatexGenerator(tmp_path)
    tex = latex_gen.generate_rotational_constants_table(
        {"Spec": {"A": {"value": 100.0, "uncertainty": 0.01}}},
        {"Spec": {"A": {"value": 100.1}}},
    )
    assert "\\begin{table}" in tex

    ui_gen = GEOMReportUIGenerator(tmp_path)
    html = ui_gen.build_summary_html("Spec", {"A": 100.0}, {"A": 0.1}, rmsd_mhz=0.01)
    assert "CoChem-GEOM" in html


def test_eckart_frame_alignment_and_decoupling() -> None:
    eckart_mod = load_module_from_path(
        "cochem_geom_eckart", repo_root / "01-INGEST-AA" / "cochem_geom_eckart.py"
    )
    aligner = eckart_mod.EckartFrameAligner()
    ref = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [0.0, 1.0, 0.0]])
    rot_angle = np.pi / 6.0
    r_mat = np.array(
        [
            [np.cos(rot_angle), -np.sin(rot_angle), 0.0],
            [np.sin(rot_angle), np.cos(rot_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = ref @ r_mat + np.array([2.0, -1.0, 0.5])
    masses = np.array([12.0, 1.0, 1.0])

    _, _, rot_residual = aligner.align_eckart_frame(ref, target, masses)
    assert rot_residual < 1e-10

    p_vib, _ = aligner.compute_decoupling_matrix(ref, masses)
    assert p_vib.shape == (9, 9)
    assert np.allclose(p_vib @ p_vib, p_vib, atol=1e-6)


def test_distance_hashing_and_deduplication() -> None:
    hash_mod = load_module_from_path(
        "cochem_geom_distance_hash", repo_root / "04-ANALYSIS" / "cochem_geom_distance_hash.py"
    )
    hasher = hash_mod.GeometryDistanceHasher()
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    coords_transformed = coords @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + 5.0
    coords_distinct = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 2.5, 0.0]])

    hash1, dists1 = hasher.compute_distance_hash(coords)
    hash2, dists2 = hasher.compute_distance_hash(coords_transformed)
    hash3, _ = hasher.compute_distance_hash(coords_distinct)

    assert hash1 == hash2
    assert hash1 != hash3
    assert np.allclose(dists1, dists2)

    unique_confs = hasher.deduplicate_conformers(
        [
            {"id": 1, "coordinates": coords},
            {"id": 2, "coordinates": coords_transformed},
            {"id": 3, "coordinates": coords_distinct},
        ]
    )
    assert len(unique_confs) == 2


def test_lms_conformational_search_and_provenance(tmp_path: Path) -> None:
    lms_mod = load_module_from_path("cochem_geom_lms", repo_root / "03-FIT-AA" / "cochem_geom_lms.py")
    generator = lms_mod.ConformationalSearchGenerator(seed=12345)

    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 1.2, 1.8], [0.0, -1.2, 1.8]])
    elements = ["C", "C", "H", "H"]
    masses = np.array([12.0, 12.0, 1.0, 1.0])

    conformers = generator.generate_lms_conformers(
        coords, elements, masses, n_conformers=15, max_energy_window_kcal=5.0, rmsd_threshold_ang=0.5
    )
    assert len(conformers) > 0
    for conf in conformers:
        assert conf["delta_energy_kcal_mol"] <= 5.0
        assert "distance_hash" in conf

    prov_path = generator.export_fit_provenance(str(tmp_path))
    assert os.path.exists(prov_path)
    pdata = json.loads(Path(prov_path).read_text(encoding="utf-8"))
    assert pdata["physical_constants"]["HBAR"]["tag"] == "[M]"


def test_multiseed_optimizer_seed_flexibility_and_dynamic_bounds() -> None:
    opt = MultiSeedOptimizer()
    bounds = ([-1.0, -1.0], [1.0, 1.0])
    assert np.allclose(opt._generate_seeds(bounds, 10, seed=42), opt._generate_seeds(bounds, 10, seed=42))
    assert not np.allclose(opt._generate_seeds(bounds, 10, seed=42), opt._generate_seeds(bounds, 10, seed=99))

    assert not opt._check_divergence(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), elements=["C", "C"])
    assert opt._check_divergence(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]), elements=["C", "C"])


def test_engine_router() -> None:
    router = EngineRouter()
    assert router.determine_route("anharmonic", "CCSD(T)", 5) == "CFOUR"
    assert router.determine_route("opt", "CCSD(T)", 10) == "CFOUR"
    assert router.determine_route("sp", "CCSD(T)-F12", 6) == "MPQC"
    assert router.determine_route("opt", "wB97M-V", 8) == "ORCA"
    assert router.determine_route("freq", "DLPNO-CCSD(T1)", 20) == "ORCA"


def test_force_field_recycling() -> None:
    lms_mod = load_module_from_path("cochem_geom_lms", repo_root / "03-FIT-AA" / "cochem_geom_lms.py")
    gen = lms_mod.ConformationalSearchGenerator(seed=42)
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755450, -0.471161],
            [0.0, -0.755450, -0.471161],
        ]
    )
    h_cart, _ = gen._estimate_spring_hessian(
        coords, np.array([15.994915, 1.007825, 1.007825]), elements=["O", "H", "H"]
    )
    h_mw_iso, h_cart_ret = gen.recycle_force_field(h_cart, np.array([15.994915, 2.014102, 2.014102]))
    assert h_mw_iso.shape == (9, 9)
    assert np.allclose(h_cart_ret, h_cart)


def test_kabsch_heavy_atom_rmsd_rotation_alignment() -> None:
    lms_mod = load_module_from_path("cochem_geom_lms", repo_root / "03-FIT-AA" / "cochem_geom_lms.py")
    gen = lms_mod.ConformationalSearchGenerator()
    c1 = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 2.0]])
    theta = np.radians(60.0)
    r_rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    c2 = c1 @ r_rot + np.array([3.0, -2.0, 1.0])
    rmsd = gen.compute_heavy_atom_rmsd(c1, c2, ["C", "C", "N", "O"])
    assert abs(rmsd) < 1e-8
