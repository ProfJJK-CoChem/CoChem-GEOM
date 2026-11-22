"""Tests for Pickett (.var, .par, .fit, .lin) and SpycFit JSON ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cochem_geom.pickett import (
    parse_pickett_fit,
    parse_pickett_lin,
    parse_pickett_par,
    parse_pickett_var,
    parse_spycfit_json,
)

def _resolve_pickett_decks_dir() -> Path:
    for candidate in (
        Path(r"D:\_orca_jobs\_publication\verification_targets\pickett_decks"),
        Path("/mnt/d/_orca_jobs/_publication/verification_targets/pickett_decks"),
    ):
        if candidate.is_dir():
            return candidate
    return Path(r"D:\_orca_jobs\_publication\verification_targets\pickett_decks")


PICKETT_DECKS_DIR = _resolve_pickett_decks_dir()
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_real_fa_dme_parent_spfit_var_exact_8_of_8() -> None:
    var_path = PICKETT_DECKS_DIR / "fa_dme_parent_spfit.var"
    assert var_path.is_file(), f"Real SPFIT .var deck missing at {var_path}"

    deck = parse_pickett_var(var_path)
    assert deck.file_kind == "var"
    assert deck.is_fitted_uncertainties is True
    assert deck.reduction == "S"
    assert deck.representation == "Ir"
    assert len(deck.parameters) == 8

    by_name = deck.by_name()
    assert set(by_name) == {"A", "B", "C", "DJ", "DJK", "DK", "d1", "d2"}

    assert by_name["A"].value_mhz == pytest.approx(5177.747506, abs=1e-5)
    assert by_name["A"].uncertainty_mhz == pytest.approx(7.180837, abs=1e-5)
    assert by_name["B"].value_mhz == pytest.approx(1824.845399, abs=1e-5)
    assert by_name["B"].uncertainty_mhz == pytest.approx(0.000824, abs=1e-5)
    assert by_name["C"].value_mhz == pytest.approx(1808.390715, abs=1e-5)
    assert by_name["C"].uncertainty_mhz == pytest.approx(0.000823, abs=1e-5)

    # Exact SPFIT quartic sign convention: raw -DJ/-DJK -> positive physical DJ/DJK
    assert by_name["DJ"].raw_spfit_value_mhz < 0.0
    assert by_name["DJ"].value_mhz == pytest.approx(0.00206905, abs=1e-7)
    assert by_name["DJ"].uncertainty_mhz == pytest.approx(0.00000686, abs=1e-7)
    assert by_name["DJK"].raw_spfit_value_mhz < 0.0
    assert by_name["DJK"].value_mhz == pytest.approx(0.00676178, abs=1e-7)
    assert by_name["DJK"].uncertainty_mhz == pytest.approx(0.00002390, abs=1e-7)
    assert by_name["DK"].is_fixed is True

    # Reconstructed correlation matrix R = L.T @ L matches SPFIT .fit pairwise correlations
    assert deck.correlation_matrix is not None
    corr = np.asarray(deck.correlation_matrix, dtype=np.float64)
    assert corr.shape == (8, 8)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-5)

    fit_summary = parse_pickett_fit(PICKETT_DECKS_DIR / "fa_dme_parent_spfit.fit")
    assert len(fit_summary.transitions) == 22
    assert fit_summary.microwave_rms_mhz == pytest.approx(0.002960, abs=1e-6)
    assert corr[0, 1] == pytest.approx(fit_summary.pairwise_correlations["1-2"], abs=1e-5)
    assert corr[1, 2] == pytest.approx(fit_summary.pairwise_correlations["2-3"], abs=1e-5)
    assert corr[1, 6] == pytest.approx(fit_summary.pairwise_correlations["2-7"], abs=1e-5)
    assert corr[0, 7] == pytest.approx(fit_summary.pairwise_correlations["1-8"], abs=1e-5)


def test_all_five_real_hcooh_dme_var_decks_and_lin_par() -> None:
    stems = [
        ("fa_dme_parent_spfit", 8, 22),
        ("h13cooh_dme", 8, 10),
        ("hcood_dme", 8, 17),
        ("dcooh_dme_2h_hfs", 10, 35),
        ("fa_13ch3och3", 8, 10),
    ]
    for stem, expected_npar, expected_nlin in stems:
        deck = parse_pickett_var(PICKETT_DECKS_DIR / f"{stem}.var")
        assert len(deck.parameters) == expected_npar
        assert deck.reduction == "S"
        assert deck.correlation_matrix is not None
        assert len(deck.correlation_matrix) == expected_npar

        lines = parse_pickett_lin(PICKETT_DECKS_DIR / f"{stem}.lin")
        assert len(lines) == expected_nlin

    # .par step limits MUST NOT be mislabelled as fitted uncertainties
    par_deck = parse_pickett_par(PICKETT_DECKS_DIR / "fa_dme_parent_spfit.par")
    assert len(par_deck.parameters) == 8
    assert par_deck.is_fitted_uncertainties is False
    assert par_deck.parameters[0].uncertainty_source == "par_prior_step_limit"
    assert par_deck.parameters[1].uncertainty_mhz == pytest.approx(100.0)


def test_line0_preserved_indented_ids_and_watson_a_representations(tmp_path: Path) -> None:
    # Compact snippet starting on line 0 with touching minus sign (regression for line-0 drop bug)
    compact = tmp_path / "compact.par"
    compact.write_text("10000 12345.6789 0.0050\n20000 54321.9876 0.0040\n", encoding="utf-8")
    deck = parse_pickett_par(compact)
    assert len(deck.parameters) == 2
    assert deck.parameters[0].name == "A"
    assert deck.parameters[0].value_mhz == pytest.approx(12345.6789)
    assert deck.parameters[1].name == "B"

    # Watson A-reduction with IIIr (oblate) representation and full SPFIT IDs
    a_var = tmp_path / "watson_a_iiir.var"
    a_var.write_text(
        "Test_Watson_A_IIIr\n"
        "   8   20    8    0     0.0000E+00     1.0000E+05     1.0000E+00 1.0000000000\n"
        "ac   1  1   0   15   0   1   1   1   0   0   0  /IIIr\n"
        "        10000   9.000000000000000E+03 1.00000000E-03 /A\n"
        "        20000   8.500000000000000E+03 1.00000000E-03 /B\n"
        "        30000   4.400000000000000E+03 1.00000000E-03 /C\n"
        "          200  -1.500000000000000E-03 1.00000000E-05 /-Delta_J\n"
        "         1100  -2.500000000000000E-03 2.00000000E-05 /-Delta_JK\n"
        "         2000  -3.500000000000000E-03 3.00000000E-05 /-Delta_K\n"
        "        40100  -4.500000000000000E-04 4.00000000E-06 /-delta_J\n"
        "        41000  -5.500000000000000E-04 5.00000000E-06 /-delta_K\n",
        encoding="utf-8",
    )
    a_deck = parse_pickett_var(a_var)
    assert a_deck.reduction == "A"
    assert a_deck.representation == "IIIr"
    a_map = a_deck.by_name()
    assert a_map["Delta_J"].value_mhz == pytest.approx(1.5e-3)
    assert a_map["Delta_JK"].value_mhz == pytest.approx(2.5e-3)
    assert a_map["Delta_K"].value_mhz == pytest.approx(3.5e-3)
    assert a_map["delta_J"].value_mhz == pytest.approx(4.5e-4)
    assert a_map["delta_K"].value_mhz == pytest.approx(5.5e-4)


def test_missing_or_zero_uncertainty_strictly_raises_value_error(tmp_path: Path) -> None:
    # 1. .var with zero uncertainty
    zero_var = tmp_path / "zero.var"
    zero_var.write_text("10000 5000.0 0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly positive"):
        parse_pickett_var(zero_var)

    # 2. .var with missing uncertainty
    missing_var = tmp_path / "missing.var"
    missing_var.write_text("10000 5000.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing uncertainty"):
        parse_pickett_var(missing_var)

    # 3. .lin with zero uncertainty
    zero_lin = tmp_path / "zero.lin"
    zero_lin.write_text("  2  1  2  1  1  1   7249.9250   0.0000   1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly positive"):
        parse_pickett_lin(zero_lin)

    # 4. SpycFit JSON with missing uncertainty
    missing_json = tmp_path / "missing.json"
    missing_json.write_text(
        json.dumps({"species_id": "H2O", "optimized_constants": {"A": {"value": 835839.1}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing 'uncertainty'"):
        parse_spycfit_json(missing_json)

    # 5. SpycFit JSON with zero uncertainty
    zero_json = tmp_path / "zero.json"
    zero_json.write_text(
        json.dumps(
            {"species_id": "H2O", "optimized_constants": {"A": {"value": 835839.1, "uncertainty": 0.0}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly positive"):
        parse_spycfit_json(zero_json)
