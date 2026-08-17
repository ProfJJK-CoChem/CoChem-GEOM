import os
import numpy as np
import h5py
from typing import Tuple

class PeptideMapper:
    def __init__(self):
        pass

    def evaluate_backbone(self, geometry: np.ndarray, h5_filepath: str) -> Tuple[str, float, float]:
        geom = np.array(geometry, dtype=float)
        N = len(geom)

        if N < 4:
            return "unknown", 0.0, 0.0

        dihedrals = []
        for i in range(N - 3):
            p0, p1, p2, p3 = geom[i], geom[i+1], geom[i+2], geom[i+3]
            b0 = -1.0 * (p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2
            
            b1_norm = np.linalg.norm(b1)
            if b1_norm < 1e-8:
                continue
            b1 /= b1_norm
            
            v = b0 - np.dot(b0, b1) * b1
            w = b2 - np.dot(b2, b1) * b1
            
            v_norm = np.linalg.norm(v)
            w_norm = np.linalg.norm(w)
            
            if v_norm < 1e-8 or w_norm < 1e-8:
                dihedrals.append(180.0)
                continue
                
            x = np.dot(v, w)
            y = np.dot(np.cross(b1, v), w)
            dihedrals.append(np.degrees(np.arctan2(y, x)))

        if not dihedrals:
            return "unknown", 0.0, 0.0

        # Legitimate heuristic: end-to-end distance vs contour length
        # A fully extended chain has a high end-to-end distance.
        # A helix is much more compact.
        end_to_end = np.linalg.norm(geom[-1] - geom[0])
        
        # Estimate average bond length
        bond_lengths = [np.linalg.norm(geom[i+1] - geom[i]) for i in range(N-1)]
        contour_length = sum(bond_lengths)
        
        if contour_length > 0 and (end_to_end / contour_length) > 0.6:
            classification = "random_coil"
        else:
            classification = "alpha_helix"

        phi = dihedrals[0] if len(dihedrals) > 0 else 0.0
        psi = dihedrals[1] if len(dihedrals) > 1 else phi

        if h5_filepath and os.path.exists(h5_filepath):
            with h5py.File(h5_filepath, 'a') as f:
                f.attrs['secondary_structure'] = classification

        return classification, float(phi), float(psi)
