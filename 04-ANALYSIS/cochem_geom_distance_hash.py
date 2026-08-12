import logging
logger = logging.getLogger(__name__)
#!/usr/bin/env python3
"""
CoChem-GEOM - Stage 4.0: 1D Interatomic Distance Matrix Hashing & Conformer Deduplication (Suggestion 44)
--------------------------------------------------------------------------------------------------
Computes rotationally and translationally invariant 1D sorted interatomic distance matrix hashes (SHA-256)
to detect topological equivalence and deduplicate molecular conformers.
"""

import hashlib
import numpy as np
from scipy.spatial.distance import pdist
from typing import Tuple, List, Dict, Any

class GeometryDistanceHasher:
    """Rotationally and translationally invariant interatomic distance matrix hasher."""

    def compute_distance_hash(
        self, coords: np.ndarray, precision_digits: int = 3
    ) -> Tuple[str, np.ndarray]:
        """
        Computes SHA-256 hash of 1D sorted interatomic distance matrix.

        Args:
            coords: Atomic Cartesian coordinates of shape (N_atoms, 3)
            precision_digits: Number of decimal places to round distances for topological invariance (default 3)

        Returns:
            Tuple (hash_digest, sorted_distances):
                - hash_digest: SHA-256 hex string representation
                - sorted_distances: Sorted 1D numpy array of upper-triangle interatomic distances
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) < 2:
            sorted_dists = np.array([0.0])
        else:
            dists = pdist(coords)
            sorted_dists = np.sort(dists)

        rounded_dists = np.round(sorted_dists, decimals=precision_digits)
        dist_str = ",".join(f"{d:.{precision_digits}f}" for d in rounded_dists)
        hash_digest = hashlib.sha256(dist_str.encode('utf-8')).hexdigest()

        return hash_digest, rounded_dists

    def _compute_valence_angles(self, coords: np.ndarray) -> np.ndarray:
        """
        Computes internal valence angles (in degrees) for atom triplets.
        Used for CREGEN angular/dihedral resolution thresholding (athr = 1.0 deg).
        """
        n_atoms = len(coords)
        if n_atoms < 3:
            return np.array([0.0])

        dists = pdist(coords)
        dist_mat = np.zeros((n_atoms, n_atoms))
        idx = 0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist_mat[i, j] = dists[idx]
                dist_mat[j, i] = dists[idx]
                idx += 1

        angles = []
        for j in range(n_atoms):
            neighbors = [i for i in range(n_atoms) if i != j and dist_mat[i, j] < 2.5]
            if len(neighbors) < 2:
                sorted_idx = np.argsort(dist_mat[j])
                neighbors = [i for i in sorted_idx if i != j][:3]

            for a in range(len(neighbors)):
                for b in range(a + 1, len(neighbors)):
                    i, k = neighbors[a], neighbors[b]
                    u = coords[i] - coords[j]
                    v = coords[k] - coords[j]
                    u_norm = np.linalg.norm(u)
                    v_norm = np.linalg.norm(v)
                    if u_norm > 1e-6 and v_norm > 1e-6:
                        cos_angle = np.clip(np.dot(u, v) / (u_norm * v_norm), -1.0, 1.0)
                        angles.append(float(np.degrees(np.arccos(cos_angle))))

        if not angles:
            return np.array([0.0])
        return np.sort(np.array(angles))

    def _compute_rotational_constants(self, coords: np.ndarray) -> np.ndarray:
        """
        Computes principal moments of inertia and rotational constants.
        Used for CREGEN rotational constant thresholding (bthr = 0.001).
        """
        n_atoms = len(coords)
        if n_atoms < 2:
            return np.array([1.0, 1.0, 1.0])

        com = np.mean(coords, axis=0)
        c_centered = coords - com
        
        I = np.zeros((3, 3))
        for r in c_centered:
            I += np.dot(r, r) * np.eye(3) - np.outer(r, r)

        evals = np.linalg.eigvalsh(I)
        evals = np.sort(evals)
        evals = np.maximum(evals, 1e-6)
        B = 1.0 / evals
        return B

    def deduplicate_conformers(
        self, conformers: List[Dict], rmsd_threshold: float = 0.05,
        angle_threshold_deg: float = 1.0, bthr: float = 0.001
    ) -> List[Dict]:
        """
        Deduplicates a list of conformer dictionaries using CREGEN / GOAT two-stage limits (§9B.3):
        - SHA-256 distance matrix hashing
        - Interatomic distance L2-norm threshold (rmsd_threshold <= 0.05 Å)
        - Valence angular/dihedral threshold (angle_threshold_deg <= 1.0 deg)
        - Rotational constant relative difference (bthr <= 0.001 = 0.1%)

        Args:
            conformer_list: List of dicts containing key "coordinates" (N_atoms, 3)
            rmsd_threshold: Maximum distance vector L2-norm difference in Å (default 0.05)
            angle_threshold_deg: Maximum angular/dihedral difference in degrees (default 1.0)
            bthr: Maximum relative difference in rotational constants B (default 0.001)

        Returns:
            Filtered list of unique conformer dictionaries.
        """
        unique_conformers = []
        seen_hashes = set()
        seen_dist_vectors = []
        seen_angle_vectors = []
        seen_rot_constants = []

        for conf in conformers:
            coords = conf.get("coordinates")
            if coords is None:
                continue

            h_digest, dist_vec = self.compute_distance_hash(coords)

            if h_digest in seen_hashes:
                continue

            angle_vec = self._compute_valence_angles(coords)
            rot_b = self._compute_rotational_constants(coords)

            is_duplicate = False
            for prev_dist, prev_angle, prev_b in zip(seen_dist_vectors, seen_angle_vectors, seen_rot_constants):
                if len(prev_dist) == len(dist_vec):
                    diff_norm = np.linalg.norm(prev_dist - dist_vec) / np.sqrt(len(dist_vec))
                    
                    if len(prev_angle) == len(angle_vec):
                        max_angle_diff = float(np.max(np.abs(prev_angle - angle_vec)))
                    else:
                        max_angle_diff = 0.0

                    rel_b_diff = float(np.max(np.abs(prev_b - rot_b) / (prev_b + 1e-12)))

                    if diff_norm < rmsd_threshold and max_angle_diff < angle_threshold_deg and rel_b_diff < bthr:
                        is_duplicate = True
                        break

            if not is_duplicate:
                seen_hashes.add(h_digest)
                seen_dist_vectors.append(dist_vec)
                seen_angle_vectors.append(angle_vec)
                seen_rot_constants.append(rot_b)
                conf_copy = conf.copy()
                conf_copy["distance_hash"] = h_digest
                unique_conformers.append(conf_copy)

        return unique_conformers
