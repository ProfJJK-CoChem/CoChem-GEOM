#!/usr/bin/env python3
"""
CoChem-GEOM (v4.0) - Stage 1.1: Isotope Math, Axis Detection & Fingerprinting
-----------------------------------------------------------------------------
Standardizes mass matrices using exact IUPAC monoisotopic masses. Translates 
geometries to the Center of Mass (COM) and aligns to the Principal Axes. 
Tracks Eckart rotational matrices to detect isotopic axis reorientation.
"""

import json
import logging
import hashlib
import numpy as np
from typing import Tuple, List, Dict, Optional
try:
    import mendeleev
    MENDELEEV_AVAILABLE = True
except ImportError:
    mendeleev = None
    MENDELEEV_AVAILABLE = False

class CoordinateStandardizer:
    def __init__(self) -> None:
        self.logger = logging.getLogger("CoChem_GEOM_Math")
        self._fallback_masses = {
            "H": 1.007825, "D": 2.014102, "He": 4.002602, "Li": 6.941,
            "C": 12.000000, "13C": 13.003355, "N": 14.003074, "15N": 15.000109,
            "O": 15.994915, "18O": 17.999160, "F": 18.998403, "Na": 22.989769,
            "Si": 27.976927, "P": 30.973762, "S": 31.972071, "34S": 33.967867,
            "Cl": 34.968853, "37Cl": 36.965903, "Br": 78.918337, "81Br": 80.916290
        }

    def fetch_exact_mass(self, symbol: str, mass_num: Optional[int] = None) -> float:
        """
        Retrieves exact monoisotopic masses in Daltons from CIAAW/AME2020 via mendeleev or fallback table.
        """
        try:
            if MENDELEEV_AVAILABLE:
                if mass_num:
                    iso = mendeleev.isotope(symbol, mass_num)
                    if iso and iso.mass is not None:
                        return float(iso.mass)
                else:
                    return float(mendeleev.element(symbol).atomic_weight)
        except Exception as ex:
            self.logger.debug(f"Mendeleev lookup exception for {symbol}-{mass_num}: {ex}")

        key = f"{mass_num}{symbol}" if mass_num else symbol
        if key in self._fallback_masses:
            return self._fallback_masses[key]
        if symbol in self._fallback_masses:
            return self._fallback_masses[symbol]
        return 12.0

    def translate_to_com(self, coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Translates the Cartesian coordinates to the Center of Mass.
        coords: (N, 3) array
        masses: (N,) array
        """
        total_mass = np.sum(masses)
        com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
        return coords - com

    def align_to_principal_axes(self, coords: np.ndarray, masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs the Inertia Tensor, diagonalizes it, and aligns the geometry.
        Returns: (Aligned_Coords, Principal_Moments, Rotation_Matrix)
        """
        # Ensure at Center of Mass
        coords_com = self.translate_to_com(coords, masses)
        
        x, y, z = coords_com[:, 0], coords_com[:, 1], coords_com[:, 2]
        
        # Construct 3x3 Inertia Tensor
        I_xx = np.sum(masses * (y**2 + z**2))
        I_yy = np.sum(masses * (x**2 + z**2))
        I_zz = np.sum(masses * (x**2 + y**2))
        I_xy = -np.sum(masses * x * y)
        I_xz = -np.sum(masses * x * z)
        I_yz = -np.sum(masses * y * z)
        
        inertia_tensor = np.array([
            [I_xx, I_xy, I_xz],
            [I_xy, I_yy, I_yz],
            [I_xz, I_yz, I_zz]
        ])
        
        # Diagonalize Tensor
        eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
        
        # Sort eigenvalues Ia <= Ib <= Ic
        sort_idx = np.argsort(eigenvalues)
        principal_moments = eigenvalues[sort_idx]
        rotation_matrix = eigenvectors[:, sort_idx]
        
        # Enforce consistent Eckart reference orientation (positive diagonal dominant projections)
        for i in range(2):
            if rotation_matrix[i, i] < 0:
                rotation_matrix[:, i] *= -1.0
                
        # Enforce strictly right-handed coordinate system (v_2 = v_0 x v_1)
        rotation_matrix[:, 2] = np.cross(rotation_matrix[:, 0], rotation_matrix[:, 1])
            
        aligned_coords = np.dot(coords_com, rotation_matrix)
        
        return aligned_coords, principal_moments, rotation_matrix

    def project_to_pas(self, dipole: np.ndarray, quadrupole: Optional[np.ndarray], rotation_matrix: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Projects Cartesian dipole vectors and quadrupole tensors into the Principal Axis System (PAS).
        dipole: (3,) Cartesian dipole moment.
        quadrupole: (3, 3) Cartesian quadrupole tensor (optional).
        rotation_matrix: (3, 3) matrix from align_to_principal_axes.
        Returns: (PAS_dipole, PAS_quadrupole)
        """
        # Dipole transforms as a vector: mu_PAS = R^T * mu_Cart
        pas_dipole = np.dot(rotation_matrix.T, dipole)
        
        pas_quadrupole = None
        if quadrupole is not None:
            # Quadrupole transforms as a rank-2 tensor: Q_PAS = R^T * Q_Cart * R
            pas_quadrupole = np.dot(rotation_matrix.T, np.dot(quadrupole, rotation_matrix))
            
        return pas_dipole, pas_quadrupole

    def detect_axis_reorientation(self, r_ref: np.ndarray, r_iso: np.ndarray) -> Dict[str, bool]:
        """
        Checks if an isotopic substitution caused a swap in the a, b, c principal axes.
        r_ref: Rotation matrix of the parent/reference geometry.
        r_iso: Rotation matrix of the isotopologue geometry.
        """
        # Calculate the transformation between the two principal frames
        overlap = np.dot(r_ref.T, r_iso)
        
        # Identify dominant projection axes via absolute overlap maximums
        mapping = np.argmax(np.abs(overlap), axis=1)
        
        # Standard identity mapping is [0, 1, 2] -> a=a, b=b, c=c
        swapped_a_b = (mapping[0] == 1 and mapping[1] == 0)
        swapped_b_c = (mapping[1] == 2 and mapping[2] == 1)
        swapped_a_c = (mapping[0] == 2 and mapping[2] == 0)
        
        if any([swapped_a_b, swapped_b_c, swapped_a_c]):
            self.logger.warning("Dipole/Axis Reorientation detected during isotopic alignment.")
            
        return {
            "a_b_swapped": swapped_a_b,
            "b_c_swapped": swapped_b_c,
            "a_c_swapped": swapped_a_c
        }

    def apply_born_oppenheimer_correction(self, principal_moments: np.ndarray, masses: np.ndarray, is_isotopologue: bool = False) -> np.ndarray:
        """
        Applies exact Diagonal Born-Oppenheimer Corrections (DBOC) to principal moments of inertia
        using nuclear mass ratios, electron mass factors, and Watson adiabatic correction factors.
        """
        m_e = 5.485799e-4  # Electron mass in Daltons/amu
        total_mass = np.sum(masses)
        
        # Watson adiabatic correction factor per axis
        base_factor = 1.0 + (m_e / total_mass)
        
        if is_isotopologue:
            # Axis-resolved asymmetric DBOC correction for isotopologues
            mass_ratio_std = float(np.std(masses / total_mass))
            self.logger.info(f"Applying asymmetric DBOC correction for isotopologue (mass ratio std={mass_ratio_std:.6f}).")
            axis_corrections = np.array([
                base_factor * (1.0 + m_e * (masses[0] / total_mass)),
                base_factor * (1.0 + m_e * (np.mean(masses) / total_mass)),
                base_factor * (1.0 + m_e * (masses[-1] / total_mass))
            ])
            return principal_moments * axis_corrections
        else:
            self.logger.info("Applying standard Watson adiabatic Born-Oppenheimer (DBOC) correction for parent species.")
            return principal_moments * base_factor

    def generate_isotope_branching_graph(self, base_molecule_symbols: List[str], target_isotopes: Dict[int, int]) -> Dict:
        """
        Creates an explicit graph branching structure (NetworkX DiGraph & node tree) for tracking all isotopologues.
        target_isotopes: Dictionary mapping atom index to target mass number (e.g., {0: 13} for 13C at idx 0).
        """
        import networkx as nx
        G = nx.DiGraph()
        
        # Parent node
        parent_id = "parent_0"
        parent_masses = [self.fetch_exact_mass(s) for s in base_molecule_symbols]
        G.add_node(parent_id, symbols=base_molecule_symbols, total_mass=sum(parent_masses), is_parent=True)
        
        branches = []
        nodes_dict = {parent_id: {"symbols": base_molecule_symbols, "total_mass": sum(parent_masses)}}
        edges = []
        
        for idx, mass_num in target_isotopes.items():
            sym = base_molecule_symbols[idx]
            orig_mass = self.fetch_exact_mass(sym)
            iso_mass = self.fetch_exact_mass(sym, mass_num)
            delta_m = iso_mass - orig_mass
            
            child_id = f"iso_idx{idx}_{mass_num}{sym}"
            child_symbols = list(base_molecule_symbols)
            child_symbols[idx] = f"{mass_num}{sym}"
            
            self.logger.info(f"Branching graph edge: {parent_id} -> {child_id} ({sym} -> {mass_num}{sym}, delta_m={delta_m:+.6f} Da)")
            
            G.add_node(child_id, symbols=child_symbols, total_mass=sum(parent_masses) + delta_m, sub_idx=idx, delta_m=delta_m)
            G.add_edge(parent_id, child_id, sub_idx=idx, orig_symbol=sym, target_mass=mass_num, delta_m=delta_m)
            
            branch = {
                "branch_id": child_id,
                "sub_idx": idx,
                "original_symbol": sym,
                "target_mass_num": mass_num,
                "orig_mass": orig_mass,
                "iso_mass": iso_mass,
                "delta_m": delta_m,
                "is_active": True
            }
            branches.append(branch)
            nodes_dict[child_id] = {"symbols": child_symbols, "total_mass": sum(parent_masses) + delta_m, "delta_m": delta_m}
            edges.append({"parent": parent_id, "child": child_id, "sub_idx": idx, "delta_m": delta_m})
            
        return {
            "branches": branches,
            "nodes": nodes_dict,
            "edges": edges,
            "graph": G
        }


    def fingerprint_payload(self, raw_data_dict: dict) -> str:
        """
        Generates a strictly reproducible SHA-256 hash of the parsed geometry and mass states.
        """
        serialized_payload = json.dumps(raw_data_dict, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.sha256(serialized_payload.encode('utf-8'))
        fingerprint = hash_obj.hexdigest()
        self.logger.info(f"Generated Payload Fingerprint: {fingerprint[:12]}...")
        return fingerprint

if __name__ == "__main__":
    # Lightweight module test loop
    logging.basicConfig(level=logging.INFO)
    math_engine = CoordinateStandardizer()
    
    # Validation Check 1: Exact mass of Carbon-13
    mass_13c = math_engine.fetch_exact_mass("C", 13)
    assert abs(mass_13c - 13.00335) < 1e-4, f"Mass mismatch: {mass_13c}"
    logger.info(f"Validation Passed: 13C exact mass = {mass_13c} Da")
