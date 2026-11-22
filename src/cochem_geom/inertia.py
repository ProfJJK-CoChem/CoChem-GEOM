"""Inertia tensor, principal axis alignment, and exact rotational g-tensor correction.

Uses the single canonical CODATA conversion constant
`ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2 = 505379.008435 MHz*u*A^2`.

Electronic contribution to moments of inertia is ONLY applied when an explicit
rotational g-tensor diagonal (g_aa, g_bb, g_cc) is supplied, using the exact
molecular rotational g-tensor relation:
    I_alpha^corr = I_alpha * (1 - (m_e / m_p) * g_{alpha,alpha})
Unphysical ad-hoc DBOC multiplications (1 +/- m_e/M) are explicitly prohibited.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from .constants import (
    ELECTRON_TO_PROTON_MASS_RATIO,
    ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2,
)


class ElectronicCorrectionResult(BaseModel):
    """Result of optional rotational g-tensor electronic correction."""

    model_config = ConfigDict(extra="forbid")

    applied: bool
    g_tensor_diagonal: tuple[float, float, float] | None = None
    corrected_moments_u_a2: tuple[float, float, float] | None = None
    corrected_rotational_constants_mhz: tuple[float, float, float] | None = None
    note: str


def translate_to_com(coords_ang: np.ndarray, masses_u: np.ndarray) -> np.ndarray:
    """Translate Cartesian coordinates (N, 3) in Angstrom to the Center of Mass."""
    coords = np.asarray(coords_ang, dtype=np.float64)
    masses = np.asarray(masses_u, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != masses.shape[0]:
        raise ValueError(
            f"Expected coords (N, 3) and masses (N,), got {coords.shape} and {masses.shape}"
        )
    if np.any(masses <= 0.0) or not np.all(np.isfinite(masses)):
        raise ValueError("All atomic masses must be finite and strictly positive.")
    total_mass = float(np.sum(masses))
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    return coords - com


def compute_inertia_tensor(coords_ang: np.ndarray, masses_u: np.ndarray) -> np.ndarray:
    """Compute the 3x3 inertia tensor (u * Angstrom^2) in the COM frame."""
    c = translate_to_com(coords_ang, masses_u)
    m = np.asarray(masses_u, dtype=np.float64)
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    i_xx = float(np.sum(m * (y * y + z * z)))
    i_yy = float(np.sum(m * (x * x + z * z)))
    i_zz = float(np.sum(m * (x * x + y * y)))
    i_xy = float(-np.sum(m * x * y))
    i_xz = float(-np.sum(m * x * z))
    i_yz = float(-np.sum(m * y * z))
    return np.array(
        [
            [i_xx, i_xy, i_xz],
            [i_xy, i_yy, i_yz],
            [i_xz, i_yz, i_zz],
        ],
        dtype=np.float64,
    )


def align_to_principal_axes(
    coords_ang: np.ndarray, masses_u: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize the inertia tensor and align coordinates to a right-handed PAS (Ia <= Ib <= Ic).

    Returns (aligned_coords_ang, principal_moments_u_a2, rotation_matrix) with det(R) == +1.
    """
    coords_com = translate_to_com(coords_ang, masses_u)
    inertia = compute_inertia_tensor(coords_com, masses_u)
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    order = np.argsort(eigenvalues)
    moments = eigenvalues[order]
    rot = eigenvectors[:, order].copy()

    for i in range(2):
        if rot[i, i] < 0.0:
            rot[:, i] *= -1.0
    rot[:, 2] = np.cross(rot[:, 0], rot[:, 1])
    norm2 = float(np.linalg.norm(rot[:, 2]))
    if norm2 > 0.0:
        rot[:, 2] /= norm2

    aligned = coords_com @ rot
    return aligned, moments, rot


def moments_to_rotational_constants_mhz(moments_u_a2: np.ndarray) -> np.ndarray:
    """Convert principal moments (Ia, Ib, Ic) [u*A^2] to (A, B, C) [MHz]."""
    m = np.asarray(moments_u_a2, dtype=np.float64)
    if m.shape != (3,) or np.any(m <= 0.0):
        raise ValueError(f"Principal moments must be a 3-vector of positive values, got {m!r}")
    return ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2 / m


def rotational_constants_to_moments_u_a2(rot_constants_mhz: np.ndarray) -> np.ndarray:
    """Convert rotational constants (A, B, C) [MHz] to principal moments (Ia, Ib, Ic) [u*A^2]."""
    b = np.asarray(rot_constants_mhz, dtype=np.float64)
    if b.shape != (3,) or np.any(b <= 0.0):
        raise ValueError(f"Rotational constants must be a 3-vector of positive values, got {b!r}")
    return ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2 / b


def rotational_constants_from_geometry(
    coords_ang: np.ndarray, masses_u: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute principal moments (Ia, Ib, Ic) [u*A^2] and (A, B, C) [MHz] from geometry."""
    _, moments, _ = align_to_principal_axes(coords_ang, masses_u)
    rot_mhz = moments_to_rotational_constants_mhz(moments)
    return moments, rot_mhz


def apply_rotational_g_tensor_correction(
    principal_moments_u_a2: np.ndarray,
    g_tensor_diagonal: tuple[float, float, float] | np.ndarray | None = None,
) -> ElectronicCorrectionResult:
    """Apply exact rotational g-tensor electronic correction when g-tensor is supplied.

    Formula (Gordy & Cook; Puzzarini, Stanton & Gauss 2010):
        I_alpha^corr = I_alpha * (1 - (m_e / m_p) * g_{alpha,alpha})

    When `g_tensor_diagonal is None`, no electronic correction is fabricated or
    applied; `applied=False` and `corrected_moments_u_a2=None` are returned.
    """
    moments = np.asarray(principal_moments_u_a2, dtype=np.float64)
    if moments.shape != (3,) or np.any(moments <= 0.0):
        raise ValueError("principal_moments_u_a2 must be a 3-vector of positive moments.")

    if g_tensor_diagonal is None:
        return ElectronicCorrectionResult(
            applied=False,
            g_tensor_diagonal=None,
            corrected_moments_u_a2=None,
            corrected_rotational_constants_mhz=None,
            note="Electronic g-tensor not supplied; electronic correction explicitly omitted (None).",
        )

    g = np.asarray(g_tensor_diagonal, dtype=np.float64)
    if g.shape != (3,) or not np.all(np.isfinite(g)):
        raise ValueError(f"g_tensor_diagonal must be 3 finite floats (g_aa, g_bb, g_cc), got {g!r}")

    corrected_moments = moments * (1.0 - ELECTRON_TO_PROTON_MASS_RATIO * g)
    if np.any(corrected_moments <= 0.0):
        raise ValueError("g-tensor correction produced non-positive moment of inertia.")
    corrected_b = moments_to_rotational_constants_mhz(corrected_moments)
    return ElectronicCorrectionResult(
        applied=True,
        g_tensor_diagonal=(float(g[0]), float(g[1]), float(g[2])),
        corrected_moments_u_a2=(
            float(corrected_moments[0]),
            float(corrected_moments[1]),
            float(corrected_moments[2]),
        ),
        corrected_rotational_constants_mhz=(
            float(corrected_b[0]),
            float(corrected_b[1]),
            float(corrected_b[2]),
        ),
        note="Exact rotational g-tensor correction I_alpha * (1 - (m_e/m_p) * g_alpha) applied.",
    )


def project_to_pas(
    dipole: np.ndarray,
    quadrupole: np.ndarray | None,
    rotation_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Project Cartesian dipole vector and optional 3x3 quadrupole tensor into PAS."""
    pas_dipole = rotation_matrix.T @ np.asarray(dipole, dtype=np.float64)
    pas_quad = None
    if quadrupole is not None:
        q = np.asarray(quadrupole, dtype=np.float64)
        pas_quad = rotation_matrix.T @ q @ rotation_matrix
    return pas_dipole, pas_quad


def detect_axis_reorientation(r_ref: np.ndarray, r_iso: np.ndarray) -> dict[str, Any]:
    """Detect whether isotopic substitution swapped principal axes relative to parent."""
    overlap = np.asarray(r_ref, dtype=np.float64).T @ np.asarray(r_iso, dtype=np.float64)
    mapping = np.argmax(np.abs(overlap), axis=1)
    perm = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        perm[i, mapping[i]] = 1.0
    return {
        "a_b_swapped": bool(mapping[0] == 1 and mapping[1] == 0),
        "b_c_swapped": bool(mapping[1] == 2 and mapping[2] == 1),
        "a_c_swapped": bool(mapping[0] == 2 and mapping[2] == 0),
        "cyclic_1": bool(mapping[0] == 1 and mapping[1] == 2 and mapping[2] == 0),
        "cyclic_2": bool(mapping[0] == 2 and mapping[1] == 0 and mapping[2] == 1),
        "permutation_matrix": perm,
        "mapping": mapping.tolist(),
    }
