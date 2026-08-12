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
try:
    import mendeleev
    MENDELEEV_AVAILABLE = True
except ImportError:
    mendeleev = None
    MENDELEEV_AVAILABLE = False
import logging
from typing import List, Dict, Tuple

class DynamicBoundsTuner:
    def __init__(self) -> None:
        self.logger = logging.getLogger("CoChem_GEOM_Bounds")
        self._radii_cache = {
            "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
            "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
            "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
            "Br": 1.20, "I": 1.39
        }

    def _get_covalent_radius(self, symbol: str) -> float:
        """Retrieves and caches covalent radii in Ångströms."""
        if symbol not in self._radii_cache:
            try:
                if MENDELEEV_AVAILABLE:
                    radius_pm = mendeleev.element(symbol).covalent_radius
                    if radius_pm:
                        self._radii_cache[symbol] = radius_pm / 100.0
                        return self._radii_cache[symbol]
            except Exception as ex:
                self.logger.debug(f"Covalent radius lookup failed for {symbol}: {ex}")
            self._radii_cache[symbol] = 1.0
        return self._radii_cache[symbol]

    def get_bond_bounds(self, atom_A: str, atom_B: str, bond_order: float = 1.0) -> Tuple[float, float]:
        """
        Calculates dynamic bounds for a bond length, incorporating bond order.
        Returns: (lower_bound_A, upper_bound_A)
        """
        r_A = self._get_covalent_radius(atom_A)
        r_B = self._get_covalent_radius(atom_B)
        
        # Pyykkö bond-order scaling factor
        bo_factor = 1.0
        if bond_order >= 3.0:
            bo_factor = 0.78
        elif bond_order >= 2.0:
            bo_factor = 0.88
        elif bond_order > 1.0:
            bo_factor = 0.94

        base_length = (r_A + r_B) * bo_factor
        
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
    def __init__(self) -> None:
        self.logger = logging.getLogger("CoChem_GEOM_ZMatrix")

    def calculate_internal_coordinates(self, coords: np.ndarray, params: List[Dict]) -> np.ndarray:
        """
        Maps a given 3N Cartesian array to the target internal coordinates.
        params: List of dictionaries defining the required internal coordinates.
        Uses smooth arctan2 cross-product angle formulations to avoid derivative singularities.
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
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 < 1e-8 or norm2 < 1e-8:
                    internals.append(0.0)
                    continue
                v1_u = v1 / norm1
                v2_u = v2 / norm2
                sin_theta = np.linalg.norm(np.cross(v1_u, v2_u))
                cos_theta = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
                internals.append(np.arctan2(sin_theta, cos_theta))
            elif p["type"] == "Dihedral":
                v1 = coords[idx[1]] - coords[idx[0]]
                v2 = coords[idx[2]] - coords[idx[1]]
                v3 = coords[idx[3]] - coords[idx[2]]
                
                n1 = np.cross(v1, v2)
                n2 = np.cross(v2, v3)
                norm1 = np.linalg.norm(n1)
                norm2 = np.linalg.norm(n2)
                if norm1 < 1e-8 or norm2 < 1e-8:
                    internals.append(0.0)
                    continue
                n1 /= norm1
                n2 /= norm2
                
                v2_u = v2 / max(np.linalg.norm(v2), 1e-8)
                m1 = np.cross(n1, v2_u)
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
        Evaluates the Wilson B-Matrix (B_ij = d q_i / d x_j) using analytical Wilson s-vectors.
        """
        num_atoms = coords.shape[0]
        num_internals = len(params)
        B_matrix = np.zeros((num_internals, num_atoms * 3))

        for i, p in enumerate(params):
            idx = p["atoms"]
            if p["type"] == "Bond":
                # s_1 = (r_1 - r_2) / |r_1 - r_2|, s_2 = -s_1
                r1, r2 = coords[idx[0]], coords[idx[1]]
                diff = r1 - r2
                dist = np.linalg.norm(diff)
                u = diff / max(dist, 1e-12)
                B_matrix[i, idx[0]*3:idx[0]*3+3] = u
                B_matrix[i, idx[1]*3:idx[1]*3+3] = -u
            elif p["type"] == "Angle":
                # Analytical s-vector for angle 1-2-3 (vertex at 2)
                r1, r2, r3 = coords[idx[0]], coords[idx[1]], coords[idx[2]]
                v1 = r1 - r2
                v2 = r3 - r2
                d1 = np.linalg.norm(v1)
                d2 = np.linalg.norm(v2)
                if d1 > 1e-10 and d2 > 1e-10:
                    u1 = v1 / d1
                    u2 = v2 / d2
                    cos_t = np.clip(np.dot(u1, u2), -1.0, 1.0)
                    sin_t = max(np.sqrt(1.0 - cos_t**2), 1e-8)
                    
                    s1 = (cos_t * u1 - u2) / (d1 * sin_t)
                    s3 = (cos_t * u2 - u1) / (d2 * sin_t)
                    s2 = -(s1 + s3)
                    
                    B_matrix[i, idx[0]*3:idx[0]*3+3] = s1
                    B_matrix[i, idx[1]*3:idx[1]*3+3] = s2
                    B_matrix[i, idx[2]*3:idx[2]*3+3] = s3
            else:
                # Fallback to high-precision central difference for dihedrals
                flat_coords = coords.flatten()
                for j in range(num_atoms * 3):
                    c_fwd = flat_coords.copy()
                    c_fwd[j] += delta
                    q_fwd = self.calculate_internal_coordinates(c_fwd.reshape(-1, 3), [p])[0]
                    c_bwd = flat_coords.copy()
                    c_bwd[j] -= delta
                    q_bwd = self.calculate_internal_coordinates(c_bwd.reshape(-1, 3), [p])[0]
                    
                    # Handle dihedral wrap [-pi, pi]
                    dq = q_fwd - q_bwd
                    if dq > np.pi: dq -= 2*np.pi
                    elif dq < -np.pi: dq += 2*np.pi
                    
                    B_matrix[i, j] = dq / (2.0 * delta)

        return B_matrix

if __name__ == "__main__":
    # Lightweight test block
    logging.basicConfig(level=logging.INFO)
    
    tuner = DynamicBoundsTuner()
    c_c_bounds = tuner.get_bond_bounds("C", "C")
    logger.info(f"Calculated C-C Dynamic Bounds: {c_c_bounds[0]:.2f} to {c_c_bounds[1]:.2f} Å")
    
    engine = ZMatrixEngine()
    # Test rigid body removal (Kabsch/Eckart)
    ref_coords = np.array([[0,0,0], [0,1,0], [1,0,0]], dtype=float)
    # Apply a distinct physical rotation and shift
    rot = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    new_coords = np.dot(ref_coords, rot) + np.array([5.0, 5.0, 5.0])
    
    aligned = engine._apply_eckart_quaternion(ref_coords, new_coords)
    logger.error(f"Alignment Error: {np.linalg.norm(aligned - ref_coords):.2e}")
