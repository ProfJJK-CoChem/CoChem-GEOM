import logging
logger = logging.getLogger(__name__)
#!/usr/bin/env python3
"""
CoChem-GEOM - Stage 3.4: Low-Mode Sampling (LMS) Conformational Search Generator (Suggestions 43 & 45)
--------------------------------------------------------------------------------------------------
Generates trial conformers along low-frequency normal mode eigenvectors, aligns geometries to Eckart frame,
performs heavy-atom Kabsch RMSD clustering (threshold <= 0.5 A), filters by energy window (delta_E <= 5.0 kcal/mol),
and serializes fit_provenance.json metadata.
"""

import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist

# Import sibling modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cochem_geom_eckart import EckartFrameAligner

try:
    from cochem_geom_distance_hash import GeometryDistanceHasher
except ImportError:
    from cochem_geom_distance_hash import GeometryDistanceHasher


COVALENT_RADII_ANG = {
    "H": 0.31, "HE": 0.28, "LI": 1.28, "BE": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "NE": 0.58, "NA": 1.66, "MG": 1.41, "AL": 1.21, "SI": 1.11, "P": 1.07, "S": 1.05, "CL": 1.02, "AR": 1.06,
    "BR": 1.20, "I": 1.39
}

class ConformationalSearchGenerator:
    """Low-Mode Sampling (LMS) conformer generator with heavy-atom RMSD clustering and energy filtering."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.eckart_aligner = EckartFrameAligner()
        self.hasher = GeometryDistanceHasher()

    def _estimate_spring_hessian(
        self, coords: np.ndarray, masses: np.ndarray, elements: Optional[List[str]] = None,
        hessian: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs Cartesian Hessian matrix H and mass-weighted Hessian matrix H_mw
        using distance-dependent covalent force constants (GEOM-04).
        If user passes explicit hessian matrix H (3N x 3N), it is used directly.
        """
        n_atoms = len(coords)
        n_dof = 3 * n_atoms
        sqrt_m = np.sqrt(masses)

        if hessian is not None:
            H = np.asarray(hessian, dtype=float)
            if H.shape != (n_dof, n_dof):
                raise ValueError(f"User Hessian shape {H.shape} does not match 3N x 3N = ({n_dof}, {n_dof})")
        else:
            H = np.zeros((n_dof, n_dof))
            # Convert 0.5 Hartree/Bohr^2 to Hartree/Angstrom^2
            k0 = 0.5 / (0.5291772109 ** 2)

            for i in range(n_atoms):
                el_i = elements[i].upper() if elements and i < len(elements) else "C"
                r_cov_i = COVALENT_RADII_ANG.get(el_i, 0.76)

                for j in range(i + 1, n_atoms):
                    el_j = elements[j].upper() if elements and j < len(elements) else "C"
                    r_cov_j = COVALENT_RADII_ANG.get(el_j, 0.76)

                    r_ij = coords[j] - coords[i]
                    d_ij = float(np.linalg.norm(r_ij))
                    if d_ij < 1e-3:
                        continue
                    u = r_ij / d_ij
                    
                    d0_ij = r_cov_i + r_cov_j
                    if d_ij <= 1.25 * d0_ij:
                        # Bonded force constant scaling: k0 * (d0 / d)^3
                        k_ij = k0 * ((d0_ij / d_ij) ** 3)
                    else:
                        # Non-bonded damped force constant
                        k_ij = 0.01 * np.exp(-2.0 * (d_ij - d0_ij))

                    K_ij = k_ij * np.outer(u, u)
                    for a in range(3):
                        for b in range(3):
                            val = K_ij[a, b]
                            H[3 * i + a, 3 * j + b] -= val
                            H[3 * j + b, 3 * i + a] -= val
                            H[3 * i + a, 3 * i + b] += val
                            H[3 * j + a, 3 * j + b] += val

        # Compute mass-weighted Hessian H_mw
        H_mw = np.zeros((n_dof, n_dof))
        for i in range(n_atoms):
            for j in range(n_atoms):
                denom = sqrt_m[i] * sqrt_m[j]
                H_mw[3*i:3*i+3, 3*j:3*j+3] = H[3*i:3*i+3, 3*j:3*j+3] / (denom if denom > 1e-6 else 1.0)

        return H, H_mw

    def compute_normal_modes(
        self, coords: np.ndarray, masses: np.ndarray, elements: Optional[List[str]] = None,
        hessian: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes projected vibrational normal mode frequencies, eigenvectors, and Cartesian Hessian H.

        Returns:
            Tuple (frequencies_cm1, normal_mode_vectors, Cartesian_Hessian_H)
        """
        P_vib, _ = self.eckart_aligner.compute_decoupling_matrix(coords, masses)
        H, H_mw = self._estimate_spring_hessian(coords, masses, elements=elements, hessian=hessian)

        # Project out rigid body translation and rotation
        H_proj = P_vib @ H_mw @ P_vib

        evals, evecs = np.linalg.eigh(H_proj)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Convert eigenvalues to approximate cm^-1 frequencies
        freqs_cm1 = np.sign(evals) * np.sqrt(np.abs(evals)) * 219474.63
        return freqs_cm1, evecs, H

    def compute_heavy_atom_rmsd(
        self, coords1: np.ndarray, coords2: np.ndarray, elements: List[str]
    ) -> float:
        """Computes heavy-atom (Z > 1) Kabsch RMSD between two geometries."""
        heavy_idx = [i for i, el in enumerate(elements) if el.upper() not in ['H', 'D', '1H', '2H']]
        if len(heavy_idx) == 0:
            heavy_idx = list(range(len(coords1)))

        c1 = coords1[heavy_idx]
        c2 = coords2[heavy_idx]

        # Center both heavy atom selections
        c1 = c1 - np.mean(c1, axis=0)
        c2 = c2 - np.mean(c2, axis=0)

        # Kabsch alignment SVD
        H = c1.T @ c2
        V, S, Wt = np.linalg.svd(H)
        d = np.linalg.det(Wt.T @ V.T)
        R_opt = Wt.T @ np.diag([1.0, 1.0, d]) @ V.T
        c2_aligned = c2 @ R_opt

        rmsd = float(np.sqrt(np.mean(np.sum((c1 - c2_aligned)**2, axis=-1))))
        return rmsd

    def generate_lms_conformers(
        self, ref_coords: np.ndarray, elements: List[str], masses: Optional[np.ndarray] = None,
        n_conformers: int = 20, max_energy_window_kcal: float = 5.0,
        rmsd_threshold_ang: float = 0.5, hessian: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Generates Low-Mode Sampling trial conformers with heavy-atom RMSD clustering and energy window filtering.
        Harmonic strain energy computed via E_strain = 0.5 * dx^T * H * dx (GEOM-04).
        """
        ref_coords = np.asarray(ref_coords, dtype=float)
        n_atoms = len(ref_coords)

        if masses is None:
            masses = np.ones(n_atoms, dtype=float)
        else:
            masses = np.asarray(masses, dtype=float)

        freqs, evecs, H = self.compute_normal_modes(ref_coords, masses, elements=elements, hessian=hessian)
        
        # Select low-frequency modes (< 300 cm^-1, ignoring first 6 zero modes)
        low_mode_idx = [i for i, f in enumerate(freqs) if 10.0 < abs(f) < 300.0]
        if len(low_mode_idx) == 0:
            low_mode_idx = list(range(6, min(12, len(freqs))))

        sqrt_m = np.sqrt(masses)
        raw_conformers = []

        for m in range(n_conformers):
            disp = np.zeros_like(ref_coords)
            for mode_i in low_mode_idx:
                mode_vec = evecs[:, mode_i].reshape((n_atoms, 3))
                amplitude = float(self.rng.normal(0.0, 0.25))
                for a in range(n_atoms):
                    disp[a] += amplitude * mode_vec[a] / max(sqrt_m[a], 1e-3)

            trial_coords = ref_coords + disp
            # Align to Eckart frame
            aligned_trial, _, rot_res = self.eckart_aligner.align_eckart_frame(ref_coords, trial_coords, masses)

            # Harmonic strain energy E_strain = 0.5 * dx^T * H * dx in kcal/mol (GEOM-04)
            dx = (aligned_trial - ref_coords).reshape(-1)
            strain_energy_kcal = float(0.5 * np.dot(dx, H @ dx) * 627.509474)

            raw_conformers.append({
                "conformer_id": m,
                "coordinates": aligned_trial,
                "relative_energy_kcal": strain_energy_kcal,
                "eckart_rot_residual": rot_res
            })

        # Sort by energy
        raw_conformers.sort(key=lambda x: x["relative_energy_kcal"])
        min_e = raw_conformers[0]["relative_energy_kcal"] if raw_conformers else 0.0

        # Energy window filter (delta_E <= 5.0 kcal/mol)
        energy_filtered = [c for c in raw_conformers if (c["relative_energy_kcal"] - min_e) <= max_energy_window_kcal]

        # Heavy-atom RMSD clustering (RMSD <= 0.5 A)
        clustered_conformers = []
        for conf in energy_filtered:
            is_duplicate = False
            for prev in clustered_conformers:
                rmsd = self.compute_heavy_atom_rmsd(conf["coordinates"], prev["coordinates"], elements)
                if rmsd <= rmsd_threshold_ang:
                    is_duplicate = True
                    break

            if not is_duplicate:
                conf["delta_energy_kcal_mol"] = round(conf["relative_energy_kcal"] - min_e, 3)
                h_digest, _ = self.hasher.compute_distance_hash(conf["coordinates"])
                conf["distance_hash"] = h_digest
                clustered_conformers.append(conf)

        return clustered_conformers

    def export_fit_provenance(self, output_dir: str = ".") -> str:
        """Serializes fit_provenance.json metadata with Section 12.5 [M]/[D]/[E] provenance tags (Suggestion 45 / GEOM-05)."""
        os.makedirs(output_dir, exist_ok=True)
        prov = {
            "module": "CoChem-GEOM",
            "seed": self.seed,
            "provenance_discipline": {
                "rule_7_compliant": True,
                "tag_schema_version": "v4.0"
            },
            "physical_constants": {
                "HBAR": {"value": 1.054571817e-34, "unit": "J*s", "tag": "[M]", "description": "Reduced Planck constant (CODATA 2018)"},
                "KB": {"value": 1.380649e-23, "unit": "J/K", "tag": "[M]", "description": "Boltzmann constant (CODATA 2018)"},
                "HARTREE_TO_KCAL_MOL": {"value": 627.509474, "unit": "kcal/mol/Eh", "tag": "[D]", "description": "Hartree to kcal/mol conversion factor"}
            },
            "structural_parameter_tags": {
                "re_equilibrium": "[D]",
                "r0_ground_state": "[D]",
                "rs_kraitchman": "[D]",
                "rm_mass_dependence": "[D]",
                "re_semi_empirical": "[D]"
            },
            "sampling_parameters": {
                "energy_window_kcal_mol": {"value": 5.0, "tag": "[E]"},
                "rmsd_threshold_angstrom": {"value": 0.5, "tag": "[E]"},
                "spring_force_constant_k0": {"value": 0.5, "tag": "[E]"}
            },
            "environment_hash": hashlib.sha256(f"CoChem-GEOM-{self.seed}".encode()).hexdigest()
        }
        out_path = os.path.join(output_dir, "fit_provenance.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prov, f, indent=4)
        return out_path
