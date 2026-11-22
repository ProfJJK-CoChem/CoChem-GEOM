"""Honest Kraitchman substitution-coordinate (r_s) solver with Costain uncertainties.

Key scientific safeguards:
- Uses single CODATA conversion factor `ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2 = 505379.008435`.
- NEVER silently clips negative r_s^2 values to 0.0: preserves exact signed `rs_squared`,
  flags `imaginary_axis=True`, sets `abs_coord_angstrom=None`, and reports
  `imaginary_magnitude_angstrom = sqrt(|rs_squared|)` with an explicit Costain warning.
- Computes Costain systematic uncertainty sigma_Costain(|r_s|) = K / |r_s| (K = 0.0015 A^2)
  alongside propagated experimental uncertainty when B0 uncertainties are provided.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from .constants import COSTAIN_CONSTANT_A2
from .inertia import rotational_constants_to_moments_u_a2


class KraitchmanAxisResult(BaseModel):
    """Per-axis Kraitchman substitution coordinate result."""

    model_config = ConfigDict(extra="forbid")

    axis: Literal["a", "b", "c"]
    delta_planar_moment_u_a2: float
    rs_squared_a2: float
    imaginary_axis: bool
    abs_coord_angstrom: float | None
    imaginary_magnitude_angstrom: float | None = None
    costain_uncertainty_angstrom: float | None = None
    experimental_uncertainty_angstrom: float | None = None
    total_uncertainty_angstrom: float | None = None
    warning: str | None = None


class KraitchmanAtomResult(BaseModel):
    """Full 3-axis Kraitchman substitution result for one isotopically substituted atom."""

    model_config = ConfigDict(extra="forbid")

    atom_label: str
    reduced_mass_u: float
    parent_moments_u_a2: tuple[float, float, float]
    isotopologue_moments_u_a2: tuple[float, float, float]
    delta_planar_moments_u_a2: tuple[float, float, float]
    condition_number: float
    any_imaginary_axis: bool
    axes: dict[Literal["a", "b", "c"], KraitchmanAxisResult]


class TriatomicRsStructure(BaseModel):
    """Substitution structure (r_s) reconstructed for a C2v triatomic XY2 (e.g. H2O)."""

    model_config = ConfigDict(extra="forbid")

    r_xy_angstrom: float
    r_xy_costain_uncertainty_angstrom: float
    angle_yxy_deg: float
    angle_yxy_costain_uncertainty_deg: float
    atom_results: dict[str, KraitchmanAtomResult]
    first_moment_residual_u_a: float
    notes: list[str]


def _planar_moments(moments_u_a2: np.ndarray) -> np.ndarray:
    """Compute planar moments P_a, P_b, P_c from principal moments I_a, I_b, I_c."""
    ia, ib, ic = float(moments_u_a2[0]), float(moments_u_a2[1]), float(moments_u_a2[2])
    return np.array(
        [
            0.5 * (-ia + ib + ic),
            0.5 * (ia - ib + ic),
            0.5 * (ia + ib - ic),
        ],
        dtype=np.float64,
    )


def _kraitchman_rs_squared(
    parent_i: np.ndarray, iso_i: np.ndarray, m_parent: float, delta_m: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute signed r_s^2 (a^2, b^2, c^2) using Kraitchman's asymmetric-top equations."""
    if m_parent <= 0.0:
        raise ValueError(f"Parent mass must be positive, got {m_parent}")
    if delta_m == 0.0 or (m_parent + delta_m) <= 0.0:
        raise ValueError(f"Invalid isotopic mass difference delta_m={delta_m} for M={m_parent}")

    mu = (m_parent * delta_m) / (m_parent + delta_m)
    p_parent = _planar_moments(parent_i)
    p_iso = _planar_moments(iso_i)
    delta_p = p_iso - p_parent

    denom_mat = np.ones((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if i != j:
                diff = float(parent_i[i] - parent_i[j])
                if abs(diff) < 1.0e-12:
                    raise ValueError(
                        f"Degenerate parent moments of inertia I_{i} == I_{j} ({parent_i[i]}); "
                        "asymmetric-top Kraitchman equations are singular for symmetric/spherical tops."
                    )
                denom_mat[i, j] = diff

    cond_k = float(np.linalg.cond(denom_mat))
    rs2 = np.zeros(3, dtype=np.float64)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        term_j = 1.0 + delta_p[j] / (parent_i[i] - parent_i[j])
        term_k = 1.0 + delta_p[k] / (parent_i[i] - parent_i[k])
        rs2[i] = (delta_p[i] / mu) * term_j * term_k

    return rs2, delta_p, mu, cond_k


def solve_kraitchman_from_moments(
    parent_moments_u_a2: np.ndarray,
    iso_moments_u_a2: np.ndarray,
    parent_mass_u: float,
    delta_mass_u: float,
    *,
    atom_label: str = "X",
    parent_moment_uncertainties_u_a2: np.ndarray | None = None,
    iso_moment_uncertainties_u_a2: np.ndarray | None = None,
    costain_k_a2: float = COSTAIN_CONSTANT_A2,
) -> KraitchmanAtomResult:
    """Solve Kraitchman's equations from principal moments of inertia (u*A^2).

    Preserves negative r_s^2 as `imaginary_axis=True` instead of silently clipping to 0.0.
    """
    p_i = np.asarray(parent_moments_u_a2, dtype=np.float64)
    i_i = np.asarray(iso_moments_u_a2, dtype=np.float64)
    if p_i.shape != (3,) or i_i.shape != (3,):
        raise ValueError("Parent and isotopologue moments must each be 3-vectors (Ia, Ib, Ic).")
    if np.any(p_i <= 0.0) or np.any(i_i <= 0.0):
        raise ValueError("All principal moments of inertia must be strictly positive.")

    rs2, delta_p, mu, cond_k = _kraitchman_rs_squared(p_i, i_i, parent_mass_u, delta_mass_u)

    # Optional experimental uncertainty propagation on r_s^2 via central finite differences
    exp_sigma_rs2: np.ndarray | None = None
    if parent_moment_uncertainties_u_a2 is not None or iso_moment_uncertainties_u_a2 is not None:
        sig_p = (
            np.asarray(parent_moment_uncertainties_u_a2, dtype=np.float64)
            if parent_moment_uncertainties_u_a2 is not None
            else np.zeros(3, dtype=np.float64)
        )
        sig_i = (
            np.asarray(iso_moment_uncertainties_u_a2, dtype=np.float64)
            if iso_moment_uncertainties_u_a2 is not None
            else np.zeros(3, dtype=np.float64)
        )
        var_rs2 = np.zeros(3, dtype=np.float64)
        for idx in range(3):
            if sig_p[idx] > 0.0:
                step = max(sig_p[idx], abs(p_i[idx]) * 1e-7)
                dp_plus = p_i.copy()
                dp_minus = p_i.copy()
                dp_plus[idx] += step
                dp_minus[idx] -= step
                r_plus, _, _, _ = _kraitchman_rs_squared(dp_plus, i_i, parent_mass_u, delta_mass_u)
                r_minus, _, _, _ = _kraitchman_rs_squared(dp_minus, i_i, parent_mass_u, delta_mass_u)
                deriv = (r_plus - r_minus) / (2.0 * step)
                var_rs2 += (deriv * sig_p[idx]) ** 2
            if sig_i[idx] > 0.0:
                step = max(sig_i[idx], abs(i_i[idx]) * 1e-7)
                di_plus = i_i.copy()
                di_minus = i_i.copy()
                di_plus[idx] += step
                di_minus[idx] -= step
                r_plus, _, _, _ = _kraitchman_rs_squared(p_i, di_plus, parent_mass_u, delta_mass_u)
                r_minus, _, _, _ = _kraitchman_rs_squared(p_i, di_minus, parent_mass_u, delta_mass_u)
                deriv = (r_plus - r_minus) / (2.0 * step)
                var_rs2 += (deriv * sig_i[idx]) ** 2
        exp_sigma_rs2 = np.sqrt(var_rs2)

    axis_names: tuple[Literal["a", "b", "c"], ...] = ("a", "b", "c")
    axes: dict[Literal["a", "b", "c"], KraitchmanAxisResult] = {}
    any_imag = False

    for idx, axis_name in enumerate(axis_names):
        val = float(rs2[idx])
        if val < 0.0:
            any_imag = True
            imag_mag = math.sqrt(abs(val))
            axes[axis_name] = KraitchmanAxisResult(
                axis=axis_name,
                delta_planar_moment_u_a2=float(delta_p[idx]),
                rs_squared_a2=val,
                imaginary_axis=True,
                abs_coord_angstrom=None,
                imaginary_magnitude_angstrom=imag_mag,
                costain_uncertainty_angstrom=None,
                experimental_uncertainty_angstrom=None,
                total_uncertainty_angstrom=None,
                warning=(
                    f"Imaginary Kraitchman coordinate on {axis_name}-axis "
                    f"(r_s^2 = {val:.6e} A^2, i*{imag_mag:.5f} A): near-axis atom or "
                    "zero-point vibrational/inertial-defect contamination (Costain limitation); "
                    "NOT clipped to 0.0."
                ),
            )
        else:
            coord = math.sqrt(val)
            costain_unc = (costain_k_a2 / coord) if coord > 1e-12 else None
            exp_unc = (
                float(exp_sigma_rs2[idx] / (2.0 * coord))
                if (exp_sigma_rs2 is not None and coord > 1e-12)
                else None
            )
            if costain_unc is not None and exp_unc is not None:
                total_unc = math.hypot(costain_unc, exp_unc)
            else:
                total_unc = costain_unc if costain_unc is not None else exp_unc

            warn = None
            if coord < 0.15:
                warn = (
                    f"Small coordinate |{axis_name}| = {coord:.5f} A (< 0.15 A): "
                    f"large Costain uncertainty ({costain_unc:.5f} A)."
                    if costain_unc is not None
                    else f"Near-zero coordinate |{axis_name}| = {coord:.5f} A."
                )

            axes[axis_name] = KraitchmanAxisResult(
                axis=axis_name,
                delta_planar_moment_u_a2=float(delta_p[idx]),
                rs_squared_a2=val,
                imaginary_axis=False,
                abs_coord_angstrom=coord,
                imaginary_magnitude_angstrom=None,
                costain_uncertainty_angstrom=costain_unc,
                experimental_uncertainty_angstrom=exp_unc,
                total_uncertainty_angstrom=total_unc,
                warning=warn,
            )

    return KraitchmanAtomResult(
        atom_label=atom_label,
        reduced_mass_u=float(mu),
        parent_moments_u_a2=(float(p_i[0]), float(p_i[1]), float(p_i[2])),
        isotopologue_moments_u_a2=(float(i_i[0]), float(i_i[1]), float(i_i[2])),
        delta_planar_moments_u_a2=(float(delta_p[0]), float(delta_p[1]), float(delta_p[2])),
        condition_number=cond_k,
        any_imaginary_axis=any_imag,
        axes=axes,
    )


def solve_kraitchman_from_rotational_constants(
    parent_abc_mhz: tuple[float, float, float] | np.ndarray,
    iso_abc_mhz: tuple[float, float, float] | np.ndarray,
    parent_mass_u: float,
    delta_mass_u: float,
    *,
    atom_label: str = "X",
    parent_unc_mhz: tuple[float, float, float] | np.ndarray | None = None,
    iso_unc_mhz: tuple[float, float, float] | np.ndarray | None = None,
) -> KraitchmanAtomResult:
    """Solve Kraitchman's equations directly from rotational constants (A, B, C) in MHz."""
    p_b = np.asarray(parent_abc_mhz, dtype=np.float64)
    i_b = np.asarray(iso_abc_mhz, dtype=np.float64)
    p_i = rotational_constants_to_moments_u_a2(p_b)
    i_i = rotational_constants_to_moments_u_a2(i_b)

    p_sig_i = None
    if parent_unc_mhz is not None:
        p_sig_b = np.asarray(parent_unc_mhz, dtype=np.float64)
        if np.any(p_sig_b <= 0.0):
            raise ValueError("Parent rotational constant uncertainties must be strictly positive.")
        p_sig_i = p_i * (p_sig_b / p_b)

    i_sig_i = None
    if iso_unc_mhz is not None:
        i_sig_b = np.asarray(iso_unc_mhz, dtype=np.float64)
        if np.any(i_sig_b <= 0.0):
            raise ValueError("Isotopologue rotational constant uncertainties must be strictly positive.")
        i_sig_i = i_i * (i_sig_b / i_b)

    return solve_kraitchman_from_moments(
        parent_moments_u_a2=p_i,
        iso_moments_u_a2=i_i,
        parent_mass_u=parent_mass_u,
        delta_mass_u=delta_mass_u,
        atom_label=atom_label,
        parent_moment_uncertainties_u_a2=p_sig_i,
        iso_moment_uncertainties_u_a2=i_sig_i,
    )


def reconstruct_c2v_triatomic_rs(
    parent_abc_mhz: tuple[float, float, float],
    parent_unc_mhz: tuple[float, float, float],
    central_sub_abc_mhz: tuple[float, float, float],
    central_sub_unc_mhz: tuple[float, float, float],
    terminal_sub_abc_mhz: tuple[float, float, float],
    terminal_sub_unc_mhz: tuple[float, float, float],
    *,
    mass_central_u: float,
    mass_central_iso_u: float,
    mass_terminal_u: float,
    mass_terminal_iso_u: float,
    central_label: str = "O",
    terminal_label: str = "H",
) -> TriatomicRsStructure:
    """Reconstruct C2v triatomic XY2 (e.g. H2O) r_s bond length & angle from HDO + H2(18O).

    In H2O PAS (Ir):
      - a-axis is parallel to H-H (central atom lies on b-axis: a_O = 0, b_O > 0, c_O = 0)
      - b-axis is the C2 symmetry axis (terminal atoms at (+-a_H, -b_H, 0))
      - c-axis is perpendicular to the molecular plane (c = 0; typically imaginary r_s^2
        due to zero-point inertial defect, preserved explicitly with imaginary_axis=True).
    """
    m_parent = mass_central_u + 2.0 * mass_terminal_u
    res_central = solve_kraitchman_from_rotational_constants(
        parent_abc_mhz,
        central_sub_abc_mhz,
        m_parent,
        mass_central_iso_u - mass_central_u,
        atom_label=central_label,
        parent_unc_mhz=parent_unc_mhz,
        iso_unc_mhz=central_sub_unc_mhz,
    )
    res_terminal = solve_kraitchman_from_rotational_constants(
        parent_abc_mhz,
        terminal_sub_abc_mhz,
        m_parent,
        mass_terminal_iso_u - mass_terminal_u,
        atom_label=terminal_label,
        parent_unc_mhz=parent_unc_mhz,
        iso_unc_mhz=terminal_sub_unc_mhz,
    )

    a_t = res_terminal.axes["a"].abs_coord_angstrom
    b_t = res_terminal.axes["b"].abs_coord_angstrom
    b_c = res_central.axes["b"].abs_coord_angstrom
    if a_t is None or b_t is None or b_c is None:
        raise ValueError(
            "In-plane Kraitchman coordinates (a_terminal, b_terminal, b_central) must be real "
            "to reconstruct C2v triatomic geometry."
        )

    # Central atom sits on opposite side of COM from the two terminal atoms along b-axis
    db = b_t + b_c
    r_xy = math.hypot(a_t, db)
    angle_rad = 2.0 * math.atan2(a_t, db)
    angle_deg = math.degrees(angle_rad)

    sig_a_t = res_terminal.axes["a"].total_uncertainty_angstrom or 0.0
    sig_b_t = res_terminal.axes["b"].total_uncertainty_angstrom or 0.0
    sig_b_c = res_central.axes["b"].total_uncertainty_angstrom or 0.0

    # Linear error propagation for r = sqrt(a_t^2 + (b_t + b_c)^2)
    dr_da = a_t / r_xy
    dr_ddb = db / r_xy
    sig_r = math.sqrt((dr_da * sig_a_t) ** 2 + (dr_ddb * sig_b_t) ** 2 + (dr_ddb * sig_b_c) ** 2)

    # Linear error propagation for theta = 2 * atan2(a_t, b_t + b_c)
    dth_da = 2.0 * db / (r_xy * r_xy)
    dth_ddb = -2.0 * a_t / (r_xy * r_xy)
    sig_th_rad = math.sqrt(
        (dth_da * sig_a_t) ** 2 + (dth_ddb * sig_b_t) ** 2 + (dth_ddb * sig_b_c) ** 2
    )
    sig_th_deg = math.degrees(sig_th_rad)

    # First-moment COM consistency check along b-axis: m_central * b_c - 2 * m_terminal * b_t
    first_moment_resid = mass_central_u * b_c - 2.0 * mass_terminal_u * b_t

    notes: list[str] = [
        f"Direct Kraitchman substitution using {central_label} ({res_central.axes['b'].abs_coord_angstrom:.5f} A on b) "
        f"and {terminal_label} (|a|={a_t:.5f} A, |b|={b_t:.5f} A).",
        f"COM first-moment residual along b-axis: {first_moment_resid:+.5f} u*A.",
    ]
    for label, atom_res in ((central_label, res_central), (terminal_label, res_terminal)):
        for ax in ("a", "b", "c"):
            w = atom_res.axes[ax].warning
            if w:
                notes.append(f"[{label}:{ax}] {w}")

    return TriatomicRsStructure(
        r_xy_angstrom=r_xy,
        r_xy_costain_uncertainty_angstrom=sig_r,
        angle_yxy_deg=angle_deg,
        angle_yxy_costain_uncertainty_deg=sig_th_deg,
        atom_results={central_label: res_central, terminal_label: res_terminal},
        first_moment_residual_u_a=first_moment_resid,
        notes=notes,
    )
