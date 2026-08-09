#!/usr/bin/env python3
"""
CoChem-GEOM (v4.0) - Stage 3.1: Internal Coordinate & Quaternion Engine
-----------------------------------------------------------------------
Mathematically bridges Cartesian space with non-redundant Z-Matrix
internal coordinates. Generates Wilson B-matrices and computes
covalent-radius-derived dynamic parameter bounds.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import mendeleev
import logging
from typing import List, Dict, Tuple

class DynamicBoundsTuner:
    def __init__(self):
        self.logger = logging.getLogger("CoChem_GEOM_Bounds")
        self._radii_cache = {}

    def _get_covalent_radius(self, symbol: str) -> float:
        """Retrieves and caches covalent radii from mendeleev in Ångströms."""
        if symbol not in self._radii_cache:
            try:
                # mendeleev returns pm, convert to Å
                radius_pm = mendeleev.element(symbol).covalent_radius
                self._radii_cache[symbol] = radius_pm / 100.0
            except Exception as e:
                self.logger.warning(f"Failed to fetch radius for {symbol}. Defaulting to 1.0 Å.")
                self._radii_cache[symbol] = 1.0
        return self._radii_cache[symbol]

    def get_bond_bounds(self, atom_A: str, atom_B: str) -> Tuple[float, float]:
        """
        Calculates dynamic bounds for a bond length.
        Returns: (lower_bound_A, upper_bound_A)
        """
        r_A = self._get_covalent_radius(atom_A)
        r_B = self._get_covalent_radius(atom_B)
        base_length = r_A + r_B
        
        # Bond limits: -20% (compression) to +25% (elongation)
        lower = max(0.5, base_length * 0.80) 
        upper = min(5.0, base_length * 1.25)
        
        return float(lower), float(upper)

    def get_angle_bounds(self) -> Tuple[float, float]:
        """Returns standard physical boundaries for angles (in radians)."""
        # 10 degrees to 178 degrees to avoid linear singularities
        return np.radians(10.0), np.radians(178.0)

    def get_dihedral_bounds(self) -> Tuple[float, float]:
        """Returns bounds for dihedral angles (in radians)."""
        return -np.pi, np.pi


class ZMatrixEngine:
    def __init__(self):
        self.logger = logging.getLogger("CoChem_GEOM_ZMatrix")

    def calculate_internal_coordinates(self, coords: np.ndarray, params: List[Dict]) -> np.ndarray:
        """
        Maps a given 3N Cartesian array to the target internal coordinates.
        params: List of dictionaries defining the required internal coordinates.
        Returns: (M,) array of internal values.
        """
        internals = []
        for p in params:
            idx = p["atoms"]
            if p["type"] == "Bond":
                v = np.linalg.norm(coords[idx[0]] - coords[idx[1]])
                internals.append(v)
            elif p["type"] == "Angle":
                v1 = coords[idx[0]] - coords[idx[1]]
                v2 = coords[idx[2]] - coords[idx[1]]
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                # Clip to prevent arccos domain errors
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                internals.append(np.arccos(cos_theta))
            elif p["type"] == "Dihedral":
                v1 = coords[idx[1]] - coords[idx[0]]
                v2 = coords[idx[2]] - coords[idx[1]]
                v3 = coords[idx[3]] - coords[idx[2]]
                
                n1 = np.cross(v1, v2)
                n2 = np.cross(v2, v3)
                n1 /= np.linalg.norm(n1)
                n2 /= np.linalg.norm(n2)
                
                m1 = np.cross(n1, v2 / np.linalg.norm(v2))
                x = np.dot(n1, n2)
                y = np.dot(m1, n2)
                internals.append(np.arctan2(y, x))
                
        return np.array(internals)

    def _apply_eckart_quaternion(self, ref_coords: np.ndarray, new_coords: np.ndarray) -> np.ndarray:
        """
        Counter-rotates the new coordinates to minimize RMSD against the reference,
        effectively preserving the Eckart frame and removing unphysical rigid-body rotation.
        """
        # Ensure both are at COM
        ref_com = np.mean(ref_coords, axis=0)
        new_com = np.mean(new_coords, axis=0)
        
        P = ref_coords - ref_com
        Q = new_coords - new_com
        
        # Kabsch SVD
        H = np.dot(Q.T, P)
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation Matrix
        R = np.dot(Vt.T, U.T)
        
        # Reflection trap
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
            
        # Optional: verify via Quaternion
        try:
            quat = Rotation.from_matrix(R)
            rotated_coords = quat.apply(Q)
            return rotated_coords
        except ValueError:
            # Fallback for ill-conditioned matrices
            self.logger.warning("Quaternion initialization failed. Using raw SVD rotation matrix.")
            return np.dot(Q, R.T)

    def compute_b_matrix(self, coords: np.ndarray, params: List[Dict], delta: float = 1e-5) -> np.ndarray:
        """
        Numerically evaluates the Wilson B-Matrix (B_ij = d q_i / d x_j).
        coords: (N, 3) Cartesian coordinates.
        params: Definition of internal variables.
        delta: Finite difference step size.
        """
        num_atoms = coords.shape[0]
        num_internals = len(params)
        B_matrix = np.zeros((num_internals, num_atoms * 3))
        
        flat_coords = coords.flatten()
        
        for j in range(num_atoms * 3):
            # Forward step
            coords_fwd = flat_coords.copy()
            coords_fwd[j] += delta
            q_fwd = self.calculate_internal_coordinates(coords_fwd.reshape(-1, 3), params)
            
            # Backward step
            coords_bwd = flat_coords.copy()
            coords_bwd[j] -= delta
            q_bwd = self.calculate_internal_coordinates(coords_bwd.reshape(-1, 3), params)
            
            # Central difference
            B_matrix[:, j] = (q_fwd - q_bwd) / (2.0 * delta)
            
        return B_matrix

if __name__ == "__main__":
    # Lightweight test block
    logging.basicConfig(level=logging.INFO)
    
    tuner = DynamicBoundsTuner()
    c_c_bounds = tuner.get_bond_bounds("C", "C")
    print(f"Calculated C-C Dynamic Bounds: {c_c_bounds[0]:.2f} to {c_c_bounds[1]:.2f} Å")
    
    engine = ZMatrixEngine()
    # Test rigid body removal (Kabsch/Eckart)
    mock_ref = np.array([[0,0,0], [0,1,0], [1,0,0]], dtype=float)
    # Apply a distinct physical rotation and shift
    rot = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    mock_new = np.dot(mock_ref, rot) + np.array([5.0, 5.0, 5.0])
    
    aligned = engine._apply_eckart_quaternion(mock_ref, mock_new)
    print(f"Alignment Error: {np.linalg.norm(aligned - mock_ref):.2e}")