"""Tests for weighted r_0 fitting, honest Kraitchman r_s, CODATA constant, and g-tensor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cochem_geom.constants import (
    ELECTRON_TO_PROTON_MASS_RATIO,
    ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2,
    get_exact_isotope_mass,
)
from cochem_geom.inertia import (
    apply_rotational_g_tensor_correction,
    moments_to_rotational_constants_mhz,
    rotational_constants_to_moments_u_a2,
)
from cochem_geom.kraitchman import (
    reconstruct_c2v_triatomic_rs,
    solve_kraitchman_from_moments,
)
from cochem_geom.r0_fitter import IsotopologueObservation, fit_c2v_triatomic_r0

REPO_ROOT = Path(__file__).resolve().parent.parent
H2O_EXP_JSON = REPO_ROOT / "benchmarks" / "h2o_t_geom_01" / "h2o_experimental_b0_piccardo2015.json"


def _load_h2o_obs() -> list[IsotopologueObservation]:
    raw = json.loads(H2O_EXP_JSON.read_text(encoding="utf-8"))
    return [IsotopologueObservation.model_validate(item) for item in raw["isotopologues"]]


def test_single_codata_rotational_constant() -> None:
    assert ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2 == pytest.approx(505379.008435, abs=1e-9)
    moments = np.array([1.0, 2.0, 3.0])
    rot = moments_to_rotational_constants_mhz(moments)
    assert np.allclose(rotational_constants_to_moments_u_a2(rot), moments, rtol=1e-12)


def test_h2o_weighted_r0_accuracy_and_weight_sensitivity() -> None:
    isos = _load_h2o_obs()
    fit_weighted = fit_c2v_triatomic_r0(isos)

    assert fit_weighted.converged is True
    assert fit_weighted.structure_type == "r0"
    assert fit_weighted.n_observations == 12
    assert fit_weighted.degrees_of_freedom == 10

    r_oh = fit_weighted.fitted_values["r(O-H)"]
    ang_hoh = fit_weighted.fitted_values["angle(H-O-H)"]

    # Published Cook/De Lucia/NIST H2O r0: r0(O-H) ~ 0.9578 A, angle0(HOH) ~ 105.0 deg
    assert abs(r_oh - 0.9578) <= 0.002
    assert abs(ang_hoh - 105.0) <= 0.2

    # Perturb weights: equalize uncertainties to 1.0 MHz -> parameters MUST change deterministically
    isos_equal = [
        IsotopologueObservation(
            species_id=iso.species_id,
            symbols=iso.symbols,
            rotational_constants_mhz=iso.rotational_constants_mhz,
            uncertainties_mhz=(1.0, 1.0, 1.0),
        )
        for iso in isos
    ]
    fit_equal_1 = fit_c2v_triatomic_r0(isos_equal)
    fit_equal_2 = fit_c2v_triatomic_r0(isos_equal)

    # Deterministic repeatability
    assert fit_equal_1.fitted_values["r(O-H)"] == pytest.approx(
        fit_equal_2.fitted_values["r(O-H)"], abs=1e-12
    )
    assert fit_equal_1.fitted_values["angle(H-O-H)"] == pytest.approx(
        fit_equal_2.fitted_values["angle(H-O-H)"], abs=1e-12
    )

    # Proves w_i = 1/sigma_i^2 is genuinely active inside scipy.optimize.least_squares
    assert abs(fit_equal_1.fitted_values["r(O-H)"] - r_oh) > 1e-4
    assert abs(fit_equal_1.fitted_values["angle(H-O-H)"] - ang_hoh) > 0.05

    # Strictly rejects zero uncertainties
    with pytest.raises(ValueError, match="strictly positive"):
        IsotopologueObservation(
            species_id="H2O",
            symbols=["16O", "1H", "1H"],
            rotational_constants_mhz=(835839.10, 435347.353, 278139.826),
            uncertainties_mhz=(0.13, 0.0, 0.057),
        )

    # Strictly refuses r_e^SE when Delta_B_vib is absent
    with pytest.raises(ValueError, match="PENDING"):
        fit_c2v_triatomic_r0(isos, structure_type="re_se")


def test_honest_kraitchman_rs_h2o_and_synthetic_negative_rs2() -> None:
    isos = {iso.species_id: iso for iso in _load_h2o_obs()}
    rs = reconstruct_c2v_triatomic_rs(
        parent_abc_mhz=isos["H2O"].rotational_constants_mhz,
        parent_unc_mhz=isos["H2O"].uncertainties_mhz,
        central_sub_abc_mhz=isos["H2(18O)"].rotational_constants_mhz,
        central_sub_unc_mhz=isos["H2(18O)"].uncertainties_mhz,
        terminal_sub_abc_mhz=isos["HDO"].rotational_constants_mhz,
        terminal_sub_unc_mhz=isos["HDO"].uncertainties_mhz,
        mass_central_u=get_exact_isotope_mass("16O"),
        mass_central_iso_u=get_exact_isotope_mass("18O"),
        mass_terminal_u=get_exact_isotope_mass("1H"),
        mass_terminal_iso_u=get_exact_isotope_mass("2H"),
    )

    # Planar H2O out-of-plane c-axis has real negative r_s^2 -> MUST be flagged imaginary_axis=True
    assert rs.atom_results["H"].axes["c"].imaginary_axis is True
    assert rs.atom_results["H"].axes["c"].rs_squared_a2 < 0.0
    assert rs.atom_results["H"].axes["c"].abs_coord_angstrom is None
    assert rs.atom_results["O"].axes["c"].imaginary_axis is True
    assert rs.atom_results["O"].axes["c"].rs_squared_a2 < 0.0
    assert rs.atom_results["O"].axes["c"].abs_coord_angstrom is None

    # In-plane coordinates and Costain uncertainties are positive and physically reasonable
    assert rs.atom_results["H"].axes["a"].costain_uncertainty_angstrom is not None
    assert rs.atom_results["H"].axes["a"].costain_uncertainty_angstrom > 0.0
    assert 0.94 <= rs.r_xy_angstrom <= 0.98
    assert 100.0 <= rs.angle_yxy_deg <= 107.0

    # Synthetic negative Delta_P_c near-axis test case
    parent_i = np.array([10.0, 50.0, 60.0])
    # Slightly decrease P_c by decreasing I_a + I_b - I_c (increase I_c slightly more than I_a+I_b)
    iso_i = np.array([10.02, 50.05, 60.09])  # dP_c = 0.5*(0.02 + 0.05 - 0.09) = -0.01 < 0
    syn = solve_kraitchman_from_moments(parent_i, iso_i, parent_mass_u=46.0, delta_mass_u=1.003355)
    assert syn.any_imaginary_axis is True
    assert syn.axes["c"].imaginary_axis is True
    assert syn.axes["c"].rs_squared_a2 < 0.0
    assert syn.axes["c"].abs_coord_angstrom is None
    assert syn.axes["c"].imaginary_magnitude_angstrom is not None
    assert syn.axes["c"].imaginary_magnitude_angstrom > 0.0


def test_exact_rotational_g_tensor_electronic_correction() -> None:
    moments = np.array([0.6046, 1.1609, 1.8170])

    # Default (no g-tensor supplied): electronic correction explicitly unset/None
    res_none = apply_rotational_g_tensor_correction(moments, g_tensor_diagonal=None)
    assert res_none.applied is False
    assert res_none.corrected_moments_u_a2 is None
    assert res_none.corrected_rotational_constants_mhz is None

    # Exact H2O experimental rotational g-tensor (Verhoeven & Dymanus 1970: g_aa~0.657, g_bb~0.718, g_cc~0.645)
    g_diag = (0.657, 0.718, 0.645)
    res_g = apply_rotational_g_tensor_correction(moments, g_tensor_diagonal=g_diag)
    assert res_g.applied is True
    assert res_g.corrected_moments_u_a2 is not None
    expected = moments * (1.0 - ELECTRON_TO_PROTON_MASS_RATIO * np.asarray(g_diag))
    assert np.allclose(res_g.corrected_moments_u_a2, expected, rtol=1e-12)
