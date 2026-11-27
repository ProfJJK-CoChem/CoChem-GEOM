"""Fail-closed Method Matrix §4.4 `%geom` / `! TightOpt` verification gate for CoChem-GEOM.

Enforces decision `sys-1126-a3d4` and CoChem Method Matrix §4.3–§4.4:
1. Any ORCA optimization deck emitting `! ... TightOpt` MUST include the complete
   five-tolerance §4.4 `%geom` convergence block (`TolE <= 1e-7`, `TolRMSG <= 3e-6`,
   `TolMaxG <= 1e-5`, `TolRMSD <= 5e-5`, `TolMaxD <= 1e-4`) paired with
   `TightSCF` or `VeryTightSCF` (and `DEFGRID3` for DFT/Hessian decks).
2. `cochem_geom.ccsd_driver.generate_orca_ccsd_input()` MUST satisfy all five §4.4
   `%geom` thresholds, `EnForceStrictConvergence true`, `VeryTightSCF`, and
   `%mdci STol <= 1e-8`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from cochem_geom.ccsd_driver import generate_orca_ccsd_input

REPO_ROOT = Path(__file__).resolve().parent.parent

SECTION_44_LIMITS: dict[str, float] = {
    "tole": 1.0e-7,
    "tolrmsg": 3.0e-6,
    "tolmaxg": 1.0e-5,
    "tolrmsd": 5.0e-5,
    "tolmaxd": 1.0e-4,
}


def validate_orca_tightopt_deck(
    deck_text: str,
    *,
    require_dft_grid: bool = False,
) -> list[str]:
    """Validate an ORCA deck containing `! TightOpt`; return violations (empty = compliant)."""
    violations: list[str] = []
    bang_lines = [
        line.strip() for line in deck_text.splitlines() if line.strip().startswith("!")
    ]
    combined_bang = " ".join(bang_lines)
    if not re.search(r"\btightopt\b", combined_bang, re.IGNORECASE):
        return violations

    if not re.search(r"\b(?:very)?tightscf\b", combined_bang, re.IGNORECASE):
        violations.append(
            "TightOpt deck missing mandatory `TightSCF` or `VeryTightSCF`."
        )

    if require_dft_grid and not re.search(r"\bdefgrid3\b", combined_bang, re.IGNORECASE):
        violations.append("DFT TightOpt deck missing mandatory `DEFGRID3`.")

    geom_match = re.search(
        r"%geom\b(.*?)\bend\b", deck_text, re.IGNORECASE | re.DOTALL
    )
    if not geom_match:
        violations.append(
            "Bare `! TightOpt` deck missing Method Matrix §4.4 `%geom` convergence block."
        )
        return violations

    geom_body = geom_match.group(1)
    for key, max_allowed in SECTION_44_LIMITS.items():
        tok = re.search(
            rf"\b{key}\s+([0-9]+(?:\.[0-9]+)?(?:[eEdD][+-]?[0-9]+)?)",
            geom_body,
            re.IGNORECASE,
        )
        if not tok:
            violations.append(
                f"Section 4.4 `%geom` block missing required `{key}` (<= {max_allowed:.1e})."
            )
            continue
        val = float(tok.group(1).replace("D", "e").replace("d", "e"))
        if val > max_allowed * (1.0 + 1e-12):
            violations.append(
                f"Section 4.4 `%geom` tolerance `{key}={val:.2e}` exceeds limit `{max_allowed:.1e}`."
            )
    return violations


def test_bare_tightopt_deck_fails_closed() -> None:
    bare_deck = """! RHF CCSD cc-pVTZ TightOpt VeryTightSCF NumGrad NumFreq
* xyz 0 1
  O 0.0 0.0 0.117
  H 0.0 0.757 -0.469
  H 0.0 -0.757 -0.469
*
"""
    violations = validate_orca_tightopt_deck(bare_deck, require_dft_grid=False)
    assert violations
    assert any("%geom" in v for v in violations)


def test_partial_geom_missing_tole_and_told_fails_closed() -> None:
    partial_deck = """! RHF CCSD cc-pVTZ TightOpt VeryTightSCF NumGrad NumFreq
%geom
  TolMaxG 5.0e-6
  TolRMSG 2.0e-6
  EnForceStrictConvergence true
end
* xyz 0 1
  O 0.0 0.0 0.117
*
"""
    violations = validate_orca_tightopt_deck(partial_deck, require_dft_grid=False)
    assert len(violations) == 3
    assert any("tole" in v.lower() for v in violations)
    assert any("tolrmsd" in v.lower() for v in violations)
    assert any("tolmaxd" in v.lower() for v in violations)


def test_generate_orca_ccsd_input_satisfies_section44_gate(tmp_path: Path) -> None:
    inp = generate_orca_ccsd_input(
        tmp_path / "h2o_ccsd.inp",
        ["O", "H", "H"],
        np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]),
        basis="cc-pVTZ",
        charge=0,
        multiplicity=1,
    )
    deck = inp.read_text(encoding="utf-8")
    assert validate_orca_tightopt_deck(deck, require_dft_grid=False) == []
    assert re.search(r"EnForceStrictConvergence\s+true", deck, re.IGNORECASE)
    assert re.search(r"STol\s+1\.0e-8", deck, re.IGNORECASE)


def test_src_cochem_geom_has_only_verified_tightopt_emitters() -> None:
    src_root = REPO_ROOT / "src" / "cochem_geom"
    allowed_emitters = {"src/cochem_geom/ccsd_driver.py"}
    unexpected: list[str] = []
    for py_file in sorted(src_root.rglob("*.py")):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        text = py_file.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), 1):
            if re.search(r"!\s*[^\n]*\btightopt\b", line, re.IGNORECASE) and rel not in allowed_emitters:
                unexpected.append(f"{rel}:{idx}: {line.strip()}")
    assert unexpected == [], f"Unexpected unverified ! TightOpt emitter in src/cochem_geom: {unexpected}"
