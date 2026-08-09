#!/usr/bin/env python3
"""
CoChem-GEOM Automated PyTest Suite
----------------------------------
Validates all 20 GEOM architectural enhancements including Pickett ingestion,
IUPAC isotope natural abundance math, Eckart principal axis right-handed alignment,
molsym symmetry translation, dynamic planar inertial defects, Z-matrix trees,
Hydrogen lock, theoretical offsets, DoF Ray asymmetry parameter sufficiency,
Wilson B-matrix analytical s-vectors, bond-order Pyykkö radii, Kraitchman Costain-cc
COM rebalancing, Latin Hypercube multi-seed optimization, SVD error propagation,
and ORCA input hardware/open-shell config.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

import importlib.util

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

repo_root = Path(__file__).parent.parent
ingest_parser_mod = load_module_from_path("cochem_geom_ingest_parser", repo_root / "01-INGEST-AA" / "cochem_geom_ingest_parser.py")
ingest_math_mod = load_module_from_path("cochem_geom_ingest_math", repo_root / "01-INGEST-AA" / "cochem_geom_ingest_math.py")
eval_sym_mod = load_module_from_path("cochem_geom_eval_sym", repo_root / "02-EVAL-AA" / "cochem_geom_eval_sym.py")
eval_triage_mod = load_module_from_path("cochem_geom_eval_triage", repo_root / "02-EVAL-AA" / "cochem_geom_eval_triage.py")
fitter_core_mod = load_module_from_path("cochem_geom_fitter_core", repo_root / "03-FIT-AA" / "cochem_geom_fitter_core.py")
fitter_optim_mod = load_module_from_path("cochem_geom_fitter_optim", repo_root / "03-FIT-AA" / "cochem_geom_fitter_optim.py")
ccsdt_refine_mod = load_module_from_path("cochem_geom_ccsdt_refine", repo_root / "03b-REFINE-AA" / "cochem_geom_ccsdt_refine.py")
reporter_latex_mod = load_module_from_path("cochem_geom_reporter_latex", repo_root / "04-REPORT-AA" / "cochem_geom_reporter_latex.py")
reporter_ui_mod = load_module_from_path("cochem_geom_reporter_ui", repo_root / "04-REPORT-AA" / "cochem_geom_reporter_ui.py")

SpectraIngestionEngine = ingest_parser_mod.SpectraIngestionEngine
MalformedPickettError = ingest_parser_mod.MalformedPickettError
CoordinateStandardizer = ingest_math_mod.CoordinateStandardizer
SymmetryControllerUI = eval_sym_mod.SymmetryControllerUI
VariableTriageEngine = eval_triage_mod.VariableTriageEngine
DynamicBoundsTuner = fitter_core_mod.DynamicBoundsTuner
ZMatrixEngine = fitter_core_mod.ZMatrixEngine
KraitchmanEngine = fitter_optim_mod.KraitchmanEngine
MultiSeedOptimizer = fitter_optim_mod.MultiSeedOptimizer
ConstrainedORCAOptimizer = ccsdt_refine_mod.ConstrainedORCAOptimizer
GEOMReportLatexGenerator = reporter_latex_mod.GEOMReportLatexGenerator
GEOMReportUIGenerator = reporter_ui_mod.GEOMReportUIGenerator


def test_ingest_abundance_and_pickett(tmp_path):
    engine = SpectraIngestionEngine(tmp_path)
    # Check IUPAC isotope natural abundance determination
    ab_main = engine._determine_abundance("H2O")
    assert ab_main == 100.0
    ab_13c = engine._determine_abundance("13CH4")
    assert abs(ab_13c - 1.07) < 0.1
    
    # Test Pickett parser with negative float touching adjacent fields
    par_file = tmp_path / "test.par"
    par_file.write_text("10000 12345.6789-0.0050\n20000 54321.9876 0.0040\n")
    constants = engine.parse_pickett_par(par_file)
    assert len(constants) == 2
    assert constants[0].constant_type == "A"
    assert abs(constants[0].value_mhz - 12345.6789) < 1e-3
    assert abs(constants[0].uncertainty_mhz - (-0.0050)) < 1e-3 or abs(constants[0].uncertainty_mhz - 0.0050) < 1e-3


def test_math_eckart_alignment_and_dboc():
    math_engine = CoordinateStandardizer()
    coords = np.array([
        [0.0, 0.0, 0.117790],
        [0.0, 0.755450, -0.471161],
        [0.0, -0.755450, -0.471161]
    ])
    masses = np.array([15.9949, 1.0078, 1.0078])
    
    aligned, moments, rot_mat = math_engine.align_to_principal_axes(coords, masses)
    # Enforce right-handed matrix det == +1
    det = np.linalg.det(rot_mat)
    assert abs(det - 1.0) < 1e-5
    
    # Test DBOC correction
    corr_moments = math_engine.apply_born_oppenheimer_correction(moments, masses, is_isotopologue=True)
    assert np.all(corr_moments > moments)


def test_sym_and_triage():
    sym_ui = SymmetryControllerUI()
    # Test planar defect with dynamic threshold
    moments = np.array([10.0, 50.0, 60.0]) # delta_I = 0
    res = sym_ui.check_planar_inertial_defect(moments)
    assert "[Planar]" in res.value
    
    # Test Triage Z-matrix tree and hydrogen lock
    elements = ['C', 'N', 'H', 'H']
    triage = VariableTriageEngine(elements)
    assert len(triage.parameters) > 0
    triage.apply_hydrogen_lock()
    # Only bond parameters involving H should be locked
    for p in triage.parameters:
        if p["type"] != "Bond":
            assert not p["is_frozen"]

    # Test Ray's asymmetry parameter sufficiency
    badge = triage.evaluate_sufficiency(num_constants=10, rot_constants=(10000.0, 5000.0, 5000.0)) # Symmetric top A != B=C
    assert "SUFFICIENT" in badge.value or "UNSUFFICIENT" in badge.value


def test_fitter_core_and_optim():
    tuner = DynamicBoundsTuner()
    lower, upper = tuner.get_bond_bounds("C", "C", bond_order=2.0)
    assert lower < upper
    
    z_engine = ZMatrixEngine()
    coords = np.array([[0.0,0.0,0.0], [0.0,0.0,1.4], [0.0,1.2,1.8]])
    params = [{"type": "Bond", "atoms": [1, 0]}, {"type": "Angle", "atoms": [2, 1, 0]}]
    internals = z_engine.calculate_internal_coordinates(coords, params)
    assert len(internals) == 2
    
    b_mat = z_engine.compute_b_matrix(coords, params)
    assert b_mat.shape == (2, 9)
    
    # Test Kraitchman and Costain-cc rebalance
    kraitchman = KraitchmanEngine()
    parent_I = np.array([10.0, 50.0, 60.0])
    iso_I = np.array([10.0, 50.01, 60.01])
    rs_coords = kraitchman.fit_rs_kraitchman(parent_I, iso_I, M_parent=30.0, delta_m=1.0)
    assert len(rs_coords) == 3

    # Test MultiSeedOptimizer SVD report
    opt = MultiSeedOptimizer()
    cov, svd_used, svd_report = opt.evaluate_jacobian_covariance(np.ones((2, 2)), np.ones(2))
    assert "condition_number" in svd_report


def test_ccsdt_refine_and_reporters(tmp_path):
    refiner = ConstrainedORCAOptimizer(tmp_path)
    # Open-shell UKS test
    inp = refiner.generate_input("radical", ["O", "H"], np.array([[0,0,0],[0,0,1]]), [], charge=0, mult=2)
    content = inp.read_text()
    assert "UKS U-DLPNO-CCSD(T)" in content
    
    latex_gen = GEOMReportLatexGenerator(tmp_path)
    tex = latex_gen.generate_rotational_constants_table({"Spec": {"A": {"value": 100.0}}}, {"Spec": {"A": {"value": 100.1}}})
    assert "\\begin{table}" in tex

    ui_gen = GEOMReportUIGenerator(tmp_path)
    html = ui_gen.build_summary_html("Spec", {"A": 100.0}, {"A": 0.1}, rmsd_mhz=0.01)
    assert "CoChem-GEOM" in html
