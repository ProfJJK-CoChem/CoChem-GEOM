import os
import logging
import numpy as np
import h5py
from pathlib import Path

logger = logging.getLogger(__name__)

class PeptideMapper:
    def __init__(self) -> None:
        """Initialize PeptideMapper with secondary structure region definitions."""
        self.regions = {
            "alpha_helix": {"phi": (-180.0, 0.0), "psi": (-100.0, 45.0)},
            "beta_sheet": {"phi": (-180.0, -45.0), "psi": (45.0, 180.0)},
            "left_handed_helix": {"phi": (0.0, 180.0), "psi": (0.0, 100.0)},
            "coil/turn": {"phi": (-180.0, 180.0), "psi": (-180.0, 180.0)}
        }

    def _compute_dihedral(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Compute the dihedral angle between four points in degrees."""
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        
        b1_norm = np.linalg.norm(b1)
        if b1_norm < 1e-8:
            return 180.0
        b1 /= b1_norm
        
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        
        v_norm = np.linalg.norm(v)
        w_norm = np.linalg.norm(w)
        
        if v_norm < 1e-8 or w_norm < 1e-8:
            return 180.0
            
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return float(np.degrees(np.arctan2(y, x)))

    def evaluate_backbone(self, geometry: np.ndarray, h5_filepath: str | Path | None) -> tuple[str, float, float]:
        geom = np.array(geometry, dtype=float)
        
        if not np.all(np.isfinite(geom)):
            raise ValueError("Geometry contains non-finite values.")
        if geom.ndim != 2 or geom.shape[1] != 3:
            raise ValueError(f"Geometry must have shape (N, 3), got {geom.shape}.")
            
        N = len(geom)
        if N < 4:
            return "unknown", 0.0, 0.0

        phis, psis, omegas = [], [], []
        
        # Assume standard backbone ordering: N, CA, C, N, CA, C...
        # Residue i starts at index 3*i
        num_residues = N // 3
        for i in range(num_residues):
            # phi_i: C(3i-1) - N(3i) - CA(3i+1) - C(3i+2)
            if i > 0 and 3*i + 2 < N:
                phis.append(self._compute_dihedral(geom[3*i - 1], geom[3*i], geom[3*i + 1], geom[3*i + 2]))
            
            # psi_i: N(3i) - CA(3i+1) - C(3i+2) - N(3i+3)
            if 3*i + 3 < N:
                psis.append(self._compute_dihedral(geom[3*i], geom[3*i + 1], geom[3*i + 2], geom[3*i + 3]))
                
            # omega_i: CA(3i+1) - C(3i+2) - N(3i+3) - CA(3i+4)
            if 3*i + 4 < N:
                omegas.append(self._compute_dihedral(geom[3*i + 1], geom[3*i + 2], geom[3*i + 3], geom[3*i + 4]))

        avg_phi = float(np.mean(phis)) if phis else 0.0
        avg_psi = float(np.mean(psis)) if psis else 0.0

        classification = "coil/turn"
        if phis or psis:
            if (-180.0 <= avg_phi <= 0.0) and (-100.0 <= avg_psi <= 45.0):
                classification = "alpha_helix"
            elif (-180.0 <= avg_phi <= -45.0) and (45.0 <= avg_psi <= 180.0):
                classification = "beta_sheet"
            elif (0.0 <= avg_phi <= 180.0) and (0.0 <= avg_psi <= 100.0):
                classification = "left_handed_helix"

        if h5_filepath and os.path.exists(h5_filepath):
            try:
                with h5py.File(h5_filepath, 'a') as f:
                    f.attrs['secondary_structure'] = classification
            except Exception as e:
                logger.error(f"Failed to write secondary structure to HDF5 file: {e}")

        return classification, avg_phi, avg_psi
