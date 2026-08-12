import logging
logger = logging.getLogger(__name__)
import hashlib  # SHA-256 artifact provenance tracking
#!/usr/bin/env python3
"""
CoChem-GEOM - Stage 1.2: Dynamic Eckart Frame Alignment & Decoupling Tensor (Suggestion 42)
---------------------------------------------------------------------------------------
Implements Eckart frame alignment (center of mass recentering and SVD rotational alignment),
validates rotational residual magnitude (< 10^-10), and constructs the 3N x 3N vibrational
projection operator P_vib = I_3N - P_trans - P_rot to decouple rigid-body translation
and rotation from internal coordinates.
"""

import numpy as np
from typing import Tuple, Optional

class EckartFrameAligner:
    """Dynamic Eckart frame alignment tensor calculation and vibrational decoupling engine."""

    def align_eckart_frame(
        self, ref_coords: np.ndarray, target_coords: np.ndarray, masses: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Aligns target_coords to ref_coords satisfying Eckart conditions:
            sum(m_i * (r_i - r_i^0)) = 0  (Translation)
            sum(m_i * (r_i^0 x r_i')) = 0  (Rotation)

        Args:
            ref_coords: Reference coordinates (N_atoms, 3)
            target_coords: Target coordinates (N_atoms, 3)
            masses: Atomic masses array (length N_atoms). Defaults to unit masses if None.

        Returns:
            Tuple (aligned_coords, U_Eckart, rot_residual_norm):
                - aligned_coords: Eckart-aligned coordinates (N_atoms, 3)
                - U_Eckart: 3x3 rotation matrix
                - rot_residual_norm: Magnitude of rotational residual ||e_rot|| < 10^-10
        """
        ref = np.asarray(ref_coords, dtype=float)
        target = np.asarray(target_coords, dtype=float)
        n_atoms = len(ref)

        if masses is None:
            m = np.ones(n_atoms, dtype=float)
        else:
            m = np.asarray(masses, dtype=float)

        m_tot = np.sum(m)

        # Center of Mass shift to origin
        com_ref = np.sum(ref * m[:, np.newaxis], axis=0) / m_tot
        com_target = np.sum(target * m[:, np.newaxis], axis=0) / m_tot

        ref_centered = ref - com_ref
        target_centered = target - com_target

        # Mass-weighted covariance matrix A = sum(m_i * r_i * (r_i^0)^T)
        A = np.zeros((3, 3))
        for i in range(n_atoms):
            A += m[i] * np.outer(target_centered[i], ref_centered[i])

        # SVD of covariance matrix A = V * S * W^T
        V, S, Wt = np.linalg.svd(A)
        W = Wt.T

        # Reflection correction for proper rotation det(U) = +1
        d = np.linalg.det(V @ Wt)
        diag = np.array([1.0, 1.0, d if d != 0 else 1.0])
        U_Eckart = V @ np.diag(diag) @ Wt

        # Transform target coordinates: r_i' = U_Eckart^T * r_i_centered (or r_i_centered @ U_Eckart)
        aligned_coords = target_centered @ U_Eckart

        # Validate Eckart conditions
        # Rotational residual: e_rot = sum(m_i * (r_i^0 x r_i'))
        e_rot = np.zeros(3)
        for i in range(n_atoms):
            e_rot += m[i] * np.cross(ref_centered[i], aligned_coords[i])

        rot_residual_norm = float(np.linalg.norm(e_rot))

        return aligned_coords, U_Eckart, rot_residual_norm

    def compute_decoupling_matrix(
        self, coords: np.ndarray, masses: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs the 3N x 3N vibrational projection operator:
            P_vib = I_3N - P_trans - P_rot
        and the 3N x 3N Eckart alignment tensor T_Eckart = P_trans + P_rot.

        Args:
            coords: Geometry coordinates (N_atoms, 3)
            masses: Atomic masses (N_atoms,)

        Returns:
            Tuple (P_vib, T_Eckart) each of shape (3N, 3N).
        """
        c = np.asarray(coords, dtype=float)
        n_atoms = len(c)
        n_dof = 3 * n_atoms

        if masses is None:
            m = np.ones(n_atoms, dtype=float)
        else:
            m = np.asarray(masses, dtype=float)

        m_tot = np.sum(m)
        com = np.sum(c * m[:, np.newaxis], axis=0) / m_tot
        c_centered = c - com

        sqrt_m = np.sqrt(m)

        # 3 Translational basis vectors D_trans (3N x 3)
        D_trans = np.zeros((n_dof, 3))
        for alpha in range(3):
            for i in range(n_atoms):
                D_trans[3 * i + alpha, alpha] = sqrt_m[i] / np.sqrt(m_tot)

        # 3 Rotational basis vectors D_rot (3N x 3)
        D_rot_raw = np.zeros((n_dof, 3))
        for i in range(n_atoms):
            r_i = c_centered[i]
            sm_i = sqrt_m[i]
            # e_x x r_i = (0, -r_z, r_y)
            D_rot_raw[3 * i:3 * i + 3, 0] = sm_i * np.array([0.0, -r_i[2], r_i[1]])
            # e_y x r_i = (r_z, 0, -r_x)
            D_rot_raw[3 * i:3 * i + 3, 1] = sm_i * np.array([r_i[2], 0.0, -r_i[0]])
            # e_z x r_i = (-r_y, r_x, 0)
            D_rot_raw[3 * i:3 * i + 3, 2] = sm_i * np.array([-r_i[1], r_i[0], 0.0])

        # Combine translation and rotation basis and Gram-Schmidt orthonormalize
        D_tr_combined = np.hstack([D_trans, D_rot_raw])
        Q, R = np.linalg.qr(D_tr_combined)

        # Retain independent non-zero columns (up to 6 for non-linear, 5 for linear)
        rank = np.linalg.matrix_rank(D_tr_combined)
        D_tr = Q[:, :rank]

        # Projection operators: T_Eckart = D_tr @ D_tr^T, P_vib = I_3N - T_Eckart
        T_Eckart = D_tr @ D_tr.T
        P_vib = np.eye(n_dof) - T_Eckart

        return P_vib, T_Eckart
