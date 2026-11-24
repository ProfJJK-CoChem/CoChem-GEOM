"""Rock-solid Pickett SPFIT (.var, .par, .fit, .lin) and CoChem-SPYFY JSON parser.

Key scientific safeguards:
- Never drops line 0 on compact parameter decks and never truncates indented
  SPFIT parameter IDs (handles full 13-character SPFIT IDs including HFS).
- Parses `.var` for true fitted rotational constants, fitted standard
  uncertainties, and reconstructs the exact parameter correlation matrix
  `R = L.T @ L` (and covariance matrix) from SPFIT's triangular Cholesky block.
- Never silently treats `.par` a priori step limits as fitted uncertainties;
  `.par` records are explicitly tagged `is_fitted_uncertainty=False`.
- Supports Watson `A` and `S` Hamiltonian reductions, `Ir`/`IIr`/`IIIr` axis
  representations, exact SPFIT parameter IDs
  (`10000, 20000, 30000, 200, 1100, 2000, 40100, 41000, 50000`), and exact
  SPFIT quartic sign conventions (`-DJ, -DJK, -DK` -> positive physical values,
  preserving raw SPFIT values alongside sign-corrected physical values).
- Strictly rejects missing, None, non-finite, or <= 0 uncertainties with
  `ValueError` (never silently substitutes `0.0`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .constants import ISOTOPE_NATURAL_ABUNDANCE_PCT

ReductionType = Literal["A", "S"]
RepresentationType = Literal["Ir", "IIr", "IIIr"]

# Canonical SPFIT parameter ID tables:
# Maps SPFIT integer ID -> (canonical_name, sign_multiplier_to_physical)
# In SPFIT, quartic centrifugal distortion terms 200, 1100, 2000 carry a minus
# sign in the Hamiltonian (-DJ, -DJK, -DK in S-reduction; -Delta_J, -Delta_JK,
# -Delta_K in A-reduction). In S-reduction, 40100 is +d1 and 50000 is +d2.
# In A-reduction, 40100 is -delta_J and 41000 is -delta_K.
SPFIT_PARAM_MAP_S: dict[int, tuple[str, float]] = {
    10000: ("A", 1.0),
    20000: ("B", 1.0),
    30000: ("C", 1.0),
    200: ("DJ", -1.0),
    1100: ("DJK", -1.0),
    2000: ("DK", -1.0),
    40100: ("d1", 1.0),
    41000: ("d1K", 1.0),
    50000: ("d2", 1.0),
    110010000: ("1.5*chi_aa", 1.0),
    110040000: ("0.25*(chi_bb-chi_cc)", 1.0),
}

SPFIT_PARAM_MAP_A: dict[int, tuple[str, float]] = {
    10000: ("A", 1.0),
    20000: ("B", 1.0),
    30000: ("C", 1.0),
    200: ("Delta_J", -1.0),
    1100: ("Delta_JK", -1.0),
    2000: ("Delta_K", -1.0),
    40100: ("delta_J", -1.0),
    41000: ("delta_K", -1.0),
    50000: ("d2", 1.0),
    110010000: ("1.5*chi_aa", 1.0),
    110040000: ("0.25*(chi_bb-chi_cc)", 1.0),
}

_FIXED_UNCERTAINTY_THRESHOLD = 1.0e-20
_FLOAT_TOKEN_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?")


class MalformedPickettError(ValueError):
    """Raised when a Pickett file cannot be parsed unambiguously."""


class PickettParameter(BaseModel):
    """Single parsed spectroscopic Hamiltonian parameter."""

    model_config = ConfigDict(extra="forbid")

    param_id: int
    name: str
    value_mhz: float
    raw_spfit_value_mhz: float
    sign_convention_factor: float
    uncertainty_mhz: float
    is_fixed: bool = False
    is_fitted_uncertainty: bool = True
    uncertainty_source: str = "var_fitted_std_err"
    comment: str = ""
    species_id: str = ""
    is_isotopologue: bool = False
    abundance_percentage: float = 100.0
    source_file: str = ""


class PickettDeck(BaseModel):
    """Complete parsed Pickett `.var` or `.par` deck with provenance and correlations."""

    model_config = ConfigDict(extra="forbid")

    title: str
    species_id: str
    source_file: str
    source_sha256: str
    file_kind: Literal["var", "par", "json"]
    reduction: ReductionType
    representation: RepresentationType
    n_parameters_declared: int | None = None
    n_lines_declared: int | None = None
    parameters: list[PickettParameter]
    correlation_matrix: list[list[float]] | None = None
    covariance_matrix: list[list[float]] | None = None
    is_fitted_uncertainties: bool = True

    def by_name(self) -> dict[str, PickettParameter]:
        return {p.name: p for p in self.parameters}

    def by_id(self) -> dict[int, PickettParameter]:
        return {p.param_id: p for p in self.parameters}

    def rotational_constants(self) -> dict[str, tuple[float, float]]:
        """Return {'A': (val, unc), 'B': (val, unc), 'C': (val, unc)}."""
        lookup = self.by_name()
        out: dict[str, tuple[float, float]] = {}
        for axis in ("A", "B", "C"):
            if axis in lookup:
                out[axis] = (lookup[axis].value_mhz, lookup[axis].uncertainty_mhz)
        return out


class PickettLine(BaseModel):
    """Single assigned transition from a Pickett `.lin` or `.fit` file."""

    model_config = ConfigDict(extra="forbid")

    line_index: int
    upper_quanta: tuple[int, ...]
    lower_quanta: tuple[int, ...]
    frequency_mhz: float
    uncertainty_mhz: float
    weight: float = 1.0
    calc_frequency_mhz: float | None = None
    residual_mhz: float | None = None


class PickettFitSummary(BaseModel):
    """Parsed summary of a Pickett SPFIT `.fit` output file."""

    model_config = ConfigDict(extra="forbid")

    title: str
    source_file: str
    source_sha256: str
    n_lines_requested: int | None = None
    n_parameters: int | None = None
    rotor_type: str | None = None
    microwave_avg_mhz: float | None = None
    microwave_rms_mhz: float | None = None
    transitions: list[PickettLine] = Field(default_factory=list)
    pairwise_correlations: dict[str, float] = Field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def determine_isotopic_abundance(species_id: str) -> float:
    """Estimate natural isotopic abundance (%) from species label."""
    matches = list(
        re.finditer(r"(?:(?P<mass>\d+))?(?P<elem>[A-Z][a-z]?|D)(?P<count>\d+)?", species_id)
    )
    if not matches:
        return 100.0
    net_abundance = 100.0
    substituted = False
    for match in matches:
        mass_str = match.group("mass") or ""
        elem = match.group("elem")
        count = int(match.group("count") or 1)
        tag = f"{mass_str}{elem}"
        if tag in ISOTOPE_NATURAL_ABUNDANCE_PCT:
            substituted = True
            net_abundance *= (ISOTOPE_NATURAL_ABUNDANCE_PCT[tag] / 100.0) ** count
    return float(net_abundance) if substituted else 100.0


def _validate_positive_uncertainty(val: Any, context: str) -> float:
    if val is None or isinstance(val, bool):
        raise ValueError(f"Missing uncertainty for {context}; silent default to 0.0 is prohibited.")
    try:
        fval = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric uncertainty {val!r} for {context}.") from exc
    if not math.isfinite(fval) or fval <= 0.0:
        raise ValueError(
            f"Uncertainty must be finite and strictly positive for {context}, got {fval!r}."
        )
    return fval


def _parse_fortran_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _parse_option_line(line: str) -> tuple[ReductionType, RepresentationType]:
    """Parse SPFIT option card for Watson reduction (A/S) and axis representation (Ir/IIr/IIIr)."""
    stripped = line.strip()
    lower = stripped.lower()

    # Check explicit representation annotations first (e.g. comments or flags)
    representation: RepresentationType = "Ir"
    if re.search(r"\biiir\b|\boblate\b", lower):
        representation = "IIIr"
    elif re.search(r"\biir\b", lower):
        representation = "IIr"
    elif re.search(r"\bir\b|\bprolate\b", lower):
        representation = "Ir"

    # Parse leading SPFIT option token (e.g. 's', 'a', 'sc', 'sb', 'ac', 'ab')
    tokens = stripped.split()
    if not tokens:
        return "A", representation

    first = tokens[0].lower()
    if first.startswith("s"):
        reduction: ReductionType = "S"
    elif first.startswith("a"):
        reduction = "A"
    else:
        raise MalformedPickettError(f"Unrecognized SPFIT Hamiltonian option line: {line!r}")

    if len(first) >= 2:
        axis_flag = first[1]
        if axis_flag == "c":
            representation = "IIIr"
        elif axis_flag == "b":
            representation = "IIr"
        elif axis_flag == "a":
            representation = "Ir"
    else:
        # In JPL calpgm SPFIT, the 6th integer (IX) or negative KMAX/ISYM can indicate
        # oblate/IIIr representation if an explicit axis letter is not given.
        ints: list[int] = []
        for tok in tokens[1:]:
            if tok.startswith("/"):
                break
            try:
                ints.append(int(tok))
            except ValueError:
                break
        # Standard prolate Ir card e.g. `s 1 1 0 15 0 1 1 1 0 0 0` has non-negative flags
        if len(ints) >= 7 and ints[6] < 0:
            representation = "IIIr"

    return reduction, representation


def _is_control_line(line: str) -> bool:
    """Detect SPFIT line 2: `NPAR NLINE NITR NXPAR MARQ MAXERR ERRTST SCALE`."""
    parts = line.strip().split()
    if len(parts) < 4:
        return False
    try:
        npar = int(parts[0])
        int(parts[1])
        int(parts[2])
        int(parts[3])
    except ValueError:
        return False
    # Control lines have small integer NPAR (typically < 1000) and at least 4 integers
    # (whereas parameter cards start with ID like 10000 or 200 followed by floats)
    if npar <= 0 or npar >= 1000:
        return False
    if len(parts) >= 5:
        try:
            _parse_fortran_float(parts[4])
            return True
        except ValueError:
            return False
    return True


def _is_option_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    first = stripped.split()[0].lower()
    return first in {"a", "s", "aa", "ab", "ac", "sa", "sb", "sc"}


def _parse_parameter_line(
    line: str,
    line_num: int,
    reduction: ReductionType,
    species_id: str,
    abundance: float,
    source_file: str,
    is_fitted_uncertainty: bool,
    uncertainty_source: str,
    allow_negative_uncertainty_abs: bool = False,
) -> PickettParameter:
    """Parse one Pickett parameter card without dropping leading spaces or truncating IDs."""
    # Split off trailing /comment cleanly
    if "/" in line:
        data_part, comment = line.split("/", 1)
        comment = comment.strip()
    else:
        data_part, comment = line, ""

    data_part = data_part.strip()
    if not data_part:
        raise MalformedPickettError(f"Empty parameter line at line {line_num + 1} in {source_file}")

    # First token MUST be integer parameter ID (up to 13+ digits, e.g. 110010000)
    id_match = re.match(r"^([+-]?\d+)\s+(.*)$", data_part)
    if not id_match:
        raise MalformedPickettError(
            f"Cannot parse SPFIT parameter ID at line {line_num + 1} in {source_file}: {line.strip()!r}"
        )

    param_id = int(id_match.group(1))
    rest = id_match.group(2)

    # Extract floats even when negative numbers touch without whitespace (12345.6789-0.0050)
    float_tokens = _FLOAT_TOKEN_RE.findall(rest)
    if len(float_tokens) < 2:
        raise ValueError(
            f"Missing uncertainty on parameter {param_id} at line {line_num + 1} in {source_file}; "
            "every parameter must provide both value and strictly positive uncertainty."
        )

    raw_val = _parse_fortran_float(float_tokens[0])
    raw_unc = _parse_fortran_float(float_tokens[1])
    if allow_negative_uncertainty_abs and raw_unc < 0.0:
        # Only for legacy compact test cards where negative prior step encodes fixed parameter
        raw_unc = abs(raw_unc)

    unc = _validate_positive_uncertainty(
        raw_unc, f"parameter {param_id} (line {line_num + 1} of {source_file})"
    )

    param_table = SPFIT_PARAM_MAP_S if reduction == "S" else SPFIT_PARAM_MAP_A
    if param_id in param_table:
        canonical_name, sign_factor = param_table[param_id]
    else:
        canonical_name = comment if comment else f"SPFIT_{param_id}"
        sign_factor = 1.0

    physical_val = sign_factor * raw_val
    # Avoid -0.0 representation
    if physical_val == 0.0:
        physical_val = 0.0

    is_fixed = unc <= _FIXED_UNCERTAINTY_THRESHOLD

    return PickettParameter(
        param_id=param_id,
        name=canonical_name,
        value_mhz=physical_val,
        raw_spfit_value_mhz=raw_val,
        sign_convention_factor=sign_factor,
        uncertainty_mhz=unc,
        is_fixed=is_fixed,
        is_fitted_uncertainty=is_fitted_uncertainty,
        uncertainty_source=uncertainty_source,
        comment=comment,
        species_id=species_id,
        is_isotopologue=(abundance < 99.0),
        abundance_percentage=abundance,
        source_file=source_file,
    )


def _parse_var_correlation_block(
    lines: list[str], n_params: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reconstruct exact correlation matrix R = L.T @ L from SPFIT `.var` triangular block.

    In JPL `calpgm` SPFIT `.var` files, the triangular block after the parameter cards
    stores `N*(N+1)/2` numbers in 10-character fixed-width columns (8 values per line).
    These entries form the lower-triangular matrix `L` (equivalently `U.T` where `U` is
    the upper-triangular Cholesky/Householder factor) such that `R = L.T @ L` has unit
    diagonal and reproduces SPFIT `.fit` pairwise correlations to full precision.
    """
    if n_params <= 0 or not lines:
        return None

    vals: list[float] = []
    for raw_line in lines:
        if not raw_line.strip():
            continue
        # Parse 10-char fixed-width fields first (handles touching minus signs cleanly)
        row_vals: list[float] = []
        if len(raw_line) >= 10:
            idx = 0
            while idx < len(raw_line):
                chunk = raw_line[idx : idx + 10].strip()
                if chunk:
                    # Guard against chunk containing two touching numbers
                    toks = _FLOAT_TOKEN_RE.findall(chunk)
                    for t in toks:
                        row_vals.append(_parse_fortran_float(t))
                idx += 10
        else:
            for t in _FLOAT_TOKEN_RE.findall(raw_line):
                row_vals.append(_parse_fortran_float(t))
        vals.extend(row_vals)

    expected = n_params * (n_params + 1) // 2
    if len(vals) != expected:
        raise MalformedPickettError(
            f"SPFIT .var correlation block has {len(vals)} entries; expected {expected} for {n_params} parameters."
        )

    lower = np.zeros((n_params, n_params), dtype=np.float64)
    cursor = 0
    for i in range(n_params):
        for j in range(i + 1):
            lower[i, j] = vals[cursor]
            cursor += 1

    corr = lower.T @ lower
    # Symmetrize floating-point rounding noise and clamp to [-1, 1]
    corr = 0.5 * (corr + corr.T)
    corr = np.clip(corr, -1.0, 1.0)
    return lower, corr


def _parse_pickett_var_or_par(
    path: Path | str,
    file_kind: Literal["var", "par"],
    default_reduction: ReductionType = "A",
    default_representation: RepresentationType = "Ir",
    allow_negative_uncertainty_abs: bool = False,
) -> PickettDeck:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Pickett file not found: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MalformedPickettError(f"Encoding error reading {p.name}: {exc}") from exc

    lines = text.splitlines()
    if not lines:
        raise MalformedPickettError(f"Empty Pickett file: {p.name}")

    species_id = p.stem
    abundance = determine_isotopic_abundance(species_id)
    digest = sha256_file(p)
    is_fitted = file_kind == "var"
    unc_source = "var_fitted_std_err" if is_fitted else "par_prior_step_limit"

    title = species_id
    reduction: ReductionType = default_reduction
    representation: RepresentationType = default_representation
    n_par_declared: int | None = None
    n_lines_declared: int | None = None

    # Determine whether this is a full SPFIT header file (title + control + option)
    # or a compact parameter-only snippet (starting directly at line 0 with param IDs).
    start_idx = 0
    if len(lines) >= 2 and _is_control_line(lines[1]):
        title = lines[0].strip() or species_id
        ctrl_parts = lines[1].strip().split()
        n_par_declared = int(ctrl_parts[0])
        n_lines_declared = int(ctrl_parts[1])
        start_idx = 2
        # Consume one or more option lines (e.g. `s 1 1 0 15 ...`)
        while start_idx < len(lines) and _is_option_line(lines[start_idx]):
            reduction, representation = _parse_option_line(lines[start_idx])
            start_idx += 1
    elif _is_option_line(lines[0]):
        reduction, representation = _parse_option_line(lines[0])
        start_idx = 1
    else:
        # Check if line 0 itself is a parameter card (must NOT be dropped!)
        start_idx = 0

    params: list[PickettParameter] = []
    corr_start_idx: int | None = None

    idx = start_idx
    while idx < len(lines):
        raw_line = lines[idx]
        if not raw_line.strip():
            idx += 1
            continue
        if raw_line.lstrip().startswith("/"):
            idx += 1
            continue
        if n_par_declared is not None and len(params) >= n_par_declared:
            corr_start_idx = idx
            break
        # Check for header-free option line if encountered before params
        if not params and _is_option_line(raw_line):
            reduction, representation = _parse_option_line(raw_line)
            idx += 1
            continue
        params.append(
            _parse_parameter_line(
                line=raw_line,
                line_num=idx,
                reduction=reduction,
                species_id=species_id,
                abundance=abundance,
                source_file=p.name,
                is_fitted_uncertainty=is_fitted,
                uncertainty_source=unc_source,
                allow_negative_uncertainty_abs=allow_negative_uncertainty_abs,
            )
        )
        idx += 1

    if not params:
        raise MalformedPickettError(f"No spectroscopic parameters parsed from {p.name}")

    if n_par_declared is not None and len(params) != n_par_declared:
        raise MalformedPickettError(
            f"Declared {n_par_declared} parameters in {p.name}, but parsed {len(params)}."
        )

    corr_list: list[list[float]] | None = None
    cov_list: list[list[float]] | None = None
    if file_kind == "var" and corr_start_idx is not None:
        parsed_corr = _parse_var_correlation_block(lines[corr_start_idx:], len(params))
        if parsed_corr is not None:
            _, corr_mat = parsed_corr
            sigmas = np.array([param.uncertainty_mhz for param in params], dtype=np.float64)
            cov_mat = corr_mat * np.outer(sigmas, sigmas)
            corr_list = corr_mat.tolist()
            cov_list = cov_mat.tolist()

    return PickettDeck(
        title=title,
        species_id=species_id,
        source_file=p.name,
        source_sha256=digest,
        file_kind=file_kind,
        reduction=reduction,
        representation=representation,
        n_parameters_declared=n_par_declared,
        n_lines_declared=n_lines_declared,
        parameters=params,
        correlation_matrix=corr_list,
        covariance_matrix=cov_list,
        is_fitted_uncertainties=is_fitted,
    )


def parse_pickett_var(path: Path | str) -> PickettDeck:
    """Parse a Pickett SPFIT `.var` file for fitted constants, uncertainties, and correlations."""
    return _parse_pickett_var_or_par(path, file_kind="var")


def parse_pickett_par(
    path: Path | str,
    *,
    allow_negative_step_abs: bool = False,
) -> PickettDeck:
    """Parse a Pickett SPFIT `.par` file (prior/step limits; NOT fitted uncertainties).

    Returned deck has `is_fitted_uncertainties=False` and each parameter carries
    `uncertainty_source='par_prior_step_limit'` so downstream fitters never confuse
    `.par` step limits (e.g. 100 MHz) with fitted experimental standard errors.
    """
    return _parse_pickett_var_or_par(
        path,
        file_kind="par",
        allow_negative_uncertainty_abs=allow_negative_step_abs,
    )


def parse_pickett_lin(path: Path | str) -> list[PickettLine]:
    """Parse a Pickett SPFIT `.lin` assigned transition list.

    Supports standard 6-quantum-number (`J Ka Kc`) and 12-quantum-number
    (hyperfine / padded `0 0 0 0 0 0`) formats. Strictly rejects missing or
    zero frequency uncertainties with `ValueError`.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Pickett .lin file not found: {p}")

    transitions: list[PickettLine] = []
    for idx, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("/"):
            continue
        data_part = stripped.split("/", 1)[0].strip()
        tokens = data_part.split()
        # Separate leading integer quantum numbers from trailing floats (freq, unc[, weight])
        first_float_idx: int | None = None
        for t_idx, tok in enumerate(tokens):
            if "." in tok or "e" in tok.lower() or "d" in tok.lower():
                first_float_idx = t_idx
                break
        if first_float_idx is None or first_float_idx < 2 or first_float_idx % 2 != 0:
            raise MalformedPickettError(
                f"Cannot identify quantum numbers and frequency at line {idx} of {p.name}: {stripped!r}"
            )
        try:
            quanta = [int(tok) for tok in tokens[:first_float_idx]]
        except ValueError as exc:
            raise MalformedPickettError(
                f"Non-integer quantum number at line {idx} of {p.name}: {stripped!r}"
            ) from exc

        float_tokens = tokens[first_float_idx:]
        if len(float_tokens) < 2:
            raise ValueError(
                f"Missing frequency uncertainty at line {idx} of {p.name}; "
                "silent default to 0.0 is prohibited."
            )

        freq = _parse_fortran_float(float_tokens[0])
        unc = _validate_positive_uncertainty(
            _parse_fortran_float(float_tokens[1]), f"transition line {idx} of {p.name}"
        )
        weight = _parse_fortran_float(float_tokens[2]) if len(float_tokens) >= 3 else 1.0
        half = len(quanta) // 2
        transitions.append(
            PickettLine(
                line_index=idx,
                upper_quanta=tuple(quanta[:half]),
                lower_quanta=tuple(quanta[half:]),
                frequency_mhz=freq,
                uncertainty_mhz=unc,
                weight=weight,
            )
        )

    if not transitions:
        raise MalformedPickettError(f"No assigned transitions found in {p.name}")
    return transitions


def parse_pickett_fit(path: Path | str) -> PickettFitSummary:
    """Parse a Pickett SPFIT `.fit` output file without misreading header blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Pickett .fit file not found: {p}")

    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise MalformedPickettError(f"Empty Pickett .fit file: {p.name}")

    title = lines[0].strip() or p.stem
    n_lines_req: int | None = None
    n_params: int | None = None
    rotor_type: str | None = None
    mw_avg: float | None = None
    mw_rms: float | None = None
    transitions: list[PickettLine] = []
    pairwise_corr: dict[str, float] = {}

    in_transition_table = False
    in_corr_section = False

    trans_re = re.compile(
        r"^\s*(\d+):\s+((?:[+-]?\d+\s+)+?)\s*([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
    )
    corr_re = re.compile(r"(\d+)\s+(\d+)\s+([+-]?\d+\.\d+)")

    for line in lines[1:]:
        stripped = line.strip()
        if "LINES REQUESTED=" in line:
            m_lines = re.search(r"LINES REQUESTED=\s*(\d+)", line)
            m_par = re.search(r"NUMBER OF PARAMETERS=\s*(\d+)", line)
            if m_lines:
                n_lines_req = int(m_lines.group(1))
            if m_par:
                n_params = int(m_par.group(1))
        elif "PROLATE ROTOR" in line:
            rotor_type = "PROLATE"
        elif "OBLATE ROTOR" in line:
            rotor_type = "OBLATE"
        elif "EXP.FREQ." in line and "CALC.FREQ." in line:
            in_transition_table = True
            in_corr_section = False
            continue
        elif "NORMALIZED DIAGONAL" in line:
            in_transition_table = False
            continue
        elif "MICROWAVE AVG =" in line:
            m_avg = re.search(r"MICROWAVE AVG\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)", line)
            if m_avg:
                mw_avg = float(m_avg.group(1))
        elif "MICROWAVE RMS =" in line:
            m_rms = re.search(r"MICROWAVE RMS\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)", line)
            if m_rms:
                mw_rms = float(m_rms.group(1))
        elif "END OF ITERATION" in line:
            in_corr_section = True
            in_transition_table = False
            continue

        if in_transition_table:
            m_t = trans_re.match(line)
            if m_t:
                line_idx = int(m_t.group(1))
                quanta = [int(q) for q in m_t.group(2).split()]
                half = len(quanta) // 2
                exp_freq = float(m_t.group(3))
                calc_freq = float(m_t.group(4))
                diff_freq = float(m_t.group(5))
                exp_err = _validate_positive_uncertainty(
                    float(m_t.group(6)), f".fit line {line_idx} in {p.name}"
                )
                transitions.append(
                    PickettLine(
                        line_index=line_idx,
                        upper_quanta=tuple(quanta[:half]),
                        lower_quanta=tuple(quanta[half:]),
                        frequency_mhz=exp_freq,
                        uncertainty_mhz=exp_err,
                        calc_frequency_mhz=calc_freq,
                        residual_mhz=diff_freq,
                    )
                )
        elif in_corr_section and stripped and not stripped.startswith(title[:10]):
            matches = corr_re.findall(line)
            if matches:
                for i_str, j_str, val_str in matches:
                    pairwise_corr[f"{int(i_str)}-{int(j_str)}"] = float(val_str)

    return PickettFitSummary(
        title=title,
        source_file=p.name,
        source_sha256=sha256_file(p),
        n_lines_requested=n_lines_req,
        n_parameters=n_params,
        rotor_type=rotor_type,
        microwave_avg_mhz=mw_avg,
        microwave_rms_mhz=mw_rms,
        transitions=transitions,
        pairwise_correlations=pairwise_corr,
    )


def parse_spyfy_json(path: Path | str) -> PickettDeck:
    """Parse a CoChem-SPYFY JSON spectroscopic constants payload.

    Strictly requires finite, strictly positive uncertainties for every constant;
    never silently substitutes 0.0 when an uncertainty is missing or zero.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"CoChem-SPYFY JSON file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted CoChem-SPYFY JSON payload {p.name}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"CoChem-SPYFY JSON payload {p.name} must be a JSON object.")

    species_id = str(data.get("species_id") or p.stem)
    reduction_raw = str(data.get("reduction", "A")).upper()
    if reduction_raw not in ("A", "S"):
        raise ValueError(f"Unsupported Hamiltonian reduction {reduction_raw!r} in {p.name}")
    reduction: ReductionType = reduction_raw  # type: ignore[assignment]

    rep_raw = str(data.get("representation", "Ir"))
    if rep_raw not in ("Ir", "IIr", "IIIr"):
        raise ValueError(f"Unsupported axis representation {rep_raw!r} in {p.name}")
    representation: RepresentationType = rep_raw  # type: ignore[assignment]

    abundance = determine_isotopic_abundance(species_id)
    opt_constants = data.get("optimized_constants")
    if not isinstance(opt_constants, dict) or not opt_constants:
        raise ValueError(f"Missing or empty 'optimized_constants' in {p.name}")

    name_to_id = {
        "A": 10000,
        "B": 20000,
        "C": 30000,
        "DJ": 200,
        "DELTA_J": 200,
        "DJK": 1100,
        "DELTA_JK": 1100,
        "DK": 2000,
        "DELTA_K": 2000,
        "D1": 40100,
        "DELTA_J_SMALL": 40100,
        "D2": 50000,
    }

    params: list[PickettParameter] = []
    for raw_name, entry in opt_constants.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Constant {raw_name!r} in {p.name} must be an object with 'value' and 'uncertainty'; "
                "bare scalars without uncertainty are prohibited."
            )
        if "value" not in entry or entry["value"] is None:
            raise ValueError(f"Missing 'value' for constant {raw_name!r} in {p.name}")
        if "uncertainty" not in entry:
            raise ValueError(
                f"Missing 'uncertainty' for constant {raw_name!r} in {p.name}; "
                "silent default to 0.0 is prohibited."
            )
        val = float(entry["value"])
        if not math.isfinite(val):
            raise ValueError(f"Non-finite value for constant {raw_name!r} in {p.name}")
        unc = _validate_positive_uncertainty(entry["uncertainty"], f"{raw_name!r} in {p.name}")
        param_id = int(entry.get("param_id", name_to_id.get(raw_name.upper(), 0)))
        params.append(
            PickettParameter(
                param_id=param_id,
                name=str(raw_name),
                value_mhz=val,
                raw_spfit_value_mhz=val,
                sign_convention_factor=1.0,
                uncertainty_mhz=unc,
                is_fixed=(unc <= _FIXED_UNCERTAINTY_THRESHOLD),
                is_fitted_uncertainty=True,
                uncertainty_source=str(entry.get("uncertainty_source", "json_standard_uncertainty")),
                species_id=species_id,
                is_isotopologue=(abundance < 99.0),
                abundance_percentage=abundance,
                source_file=p.name,
            )
        )

    return PickettDeck(
        title=str(data.get("title", species_id)),
        species_id=species_id,
        source_file=p.name,
        source_sha256=sha256_file(p),
        file_kind="json",
        reduction=reduction,
        representation=representation,
        n_parameters_declared=len(params),
        parameters=params,
        correlation_matrix=data.get("correlation_matrix"),
        covariance_matrix=data.get("covariance_matrix"),
        is_fitted_uncertainties=True,
    )
