"""Zero-mock tests verifying Stage 2.1 symmetry detection and C1 fallback warnings."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _load_eval_sym() -> Any:
    repo_root = Path(__file__).resolve().parent.parent
    mod_path = repo_root / "02-EVAL-AA" / "cochem_geom_eval_sym.py"
    spec = importlib.util.spec_from_file_location("cochem_geom_eval_sym_warn_test", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cochem_geom_eval_sym_warn_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _flatten_widget_html(widget: Any) -> str:
    parts: list[str] = []
    if hasattr(widget, "value") and isinstance(widget.value, str):
        parts.append(widget.value)
    if hasattr(widget, "children"):
        for child in widget.children:
            parts.append(_flatten_widget_html(child))
    return "\n".join(parts)


def test_c1_fallback_emits_user_warning_when_molsym_unavailable() -> None:
    eval_sym = _load_eval_sym()
    sym_ui = eval_sym.SymmetryControllerUI(use_molsym=False)
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755450, -0.471161],
            [0.0, -0.755450, -0.471161],
        ]
    )

    with pytest.warns(UserWarning, match=r"unverified point group 'C1'"):
        pg = sym_ui.analyze_symmetry(coords, ["O", "H", "H"], "Iso_H2O")

    assert pg == "C1"
    assert sym_ui.last_symmetry_warning is not None
    assert "Iso_H2O" in sym_ui.last_symmetry_warning
    assert "unverified point group 'C1'" in sym_ui.last_symmetry_warning
    assert sym_ui.symmetry_warnings["Iso_H2O"] == sym_ui.last_symmetry_warning


def test_c1_fallback_emits_user_warning_on_real_molsym_failure() -> None:
    eval_sym = _load_eval_sym()
    sym_ui = eval_sym.SymmetryControllerUI()
    # Non-finite coordinate triggers genuine failure inside analyze_symmetry / molsym
    bad_coords = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]])

    with pytest.warns(UserWarning, match=r"unverified point group 'C1'"):
        pg = sym_ui.analyze_symmetry(bad_coords, ["O", "H"], "Iso_BadCoords")

    assert pg == "C1"
    assert sym_ui.last_symmetry_warning is not None
    assert "Iso_BadCoords" in sym_ui.last_symmetry_warning

    # Also test genuine molsym NotAnElementError with unknown atomic symbol when molsym is installed
    if eval_sym.MOLSYM_AVAILABLE:
        valid_shape = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.warns(UserWarning, match=r"molsym detection failed"):
            pg_bad_el = sym_ui.analyze_symmetry(valid_shape, ["Xx", "H"], "Iso_BadElem")
        assert pg_bad_el == "C1"
        assert "Iso_BadElem" in sym_ui.symmetry_warnings


def test_hitl_dashboard_renders_visible_c1_fallback_banner() -> None:
    eval_sym = _load_eval_sym()
    sym_ui = eval_sym.SymmetryControllerUI(use_molsym=False)
    payload = {
        "Iso_001_Water": {
            "coords": np.array(
                [
                    [0.0, 0.0, 0.117790],
                    [0.0, 0.755450, -0.471161],
                    [0.0, -0.755450, -0.471161],
                ]
            ),
            "elements": ["O", "H", "H"],
            "principal_moments": np.array([0.5, 1.0, 1.4]),
        }
    }

    with pytest.warns(UserWarning, match=r"unverified point group 'C1'"):
        panel = sym_ui.render_hitl_dashboard(payload)

    rendered_html = _flatten_widget_html(panel)
    assert "Symmetry Fallback Warning" in rendered_html
    assert "Unverified C1 Fallback" in rendered_html
    assert "Iso_001_Water" in rendered_html
    assert sym_ui.symmetry_override_dict["Iso_001_Water"] == "C1"


def test_valid_h2o_detects_c2v_without_fallback_warning_when_molsym_installed() -> None:
    eval_sym = _load_eval_sym()
    if not eval_sym.MOLSYM_AVAILABLE:
        pytest.skip("molsym not installed in environment")

    sym_ui = eval_sym.SymmetryControllerUI()
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755450, -0.471161],
            [0.0, -0.755450, -0.471161],
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pg = sym_ui.analyze_symmetry(coords, ["O", "H", "H"], "H2O_Valid")

    assert pg.lower() == "c2v"
    assert sym_ui.last_symmetry_warning is None
    assert "H2O_Valid" not in sym_ui.symmetry_warnings
    assert len(caught) == 0
