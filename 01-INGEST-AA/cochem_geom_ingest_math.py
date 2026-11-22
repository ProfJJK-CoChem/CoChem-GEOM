#!/usr/bin/env python3
"""CoChem-GEOM Stage 1.1: Isotope Math, PAS Alignment & Rotational g-Tensor Correction.

Delegates to `cochem_geom.constants` and `cochem_geom.inertia` using the single
canonical CODATA conversion constant `505379.008435 MHz*u*A^2` and exact
rotational g-tensor electronic correction (removing unphysical ad-hoc DBOC).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np

from cochem_geom.constants import (
    ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2,
    get_exact_isotope_mass,
)
from cochem_geom.inertia import (
    ElectronicCorrectionResult,
    align_to_principal_axes,
    apply_rotational_g_tensor_correction,
    detect_axis_reorientation,
    project_to_pas,
    translate_to_com,
)


class CoordinateStandardizer:
    ROTATIONAL_CONVERSION_MHZ_U_A2 = ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2

    def __init__(self) -> None:
        self.logger = logging.getLogger("CoChem_GEOM_Math")

    def fetch_exact_mass(self, symbol: str, mass_num: int | None = None) -> float:
        return get_exact_isotope_mass(symbol, mass_num)

    def translate_to_com(self, coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
        return translate_to_com(coords, masses)

    def align_to_principal_axes(
        self, coords: np.ndarray, masses: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return align_to_principal_axes(coords, masses)

    def project_to_pas(
        self,
        dipole: np.ndarray,
        quadrupole: np.ndarray | None,
        rotation_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return project_to_pas(dipole, quadrupole, rotation_matrix)

    def detect_axis_reorientation(self, r_ref: np.ndarray, r_iso: np.ndarray) -> dict[str, Any]:
        return detect_axis_reorientation(r_ref, r_iso)

    def apply_electronic_g_tensor_correction(
        self,
        principal_moments: np.ndarray,
        g_tensor_diagonal: tuple[float, float, float] | np.ndarray | None = None,
    ) -> ElectronicCorrectionResult:
        """Apply exact I_alpha * (1 - (m_e/m_p) * g_{alpha,alpha}) only when g_tensor is supplied."""
        return apply_rotational_g_tensor_correction(principal_moments, g_tensor_diagonal)

    def generate_isotope_branching_graph(
        self, base_molecule_symbols: list[str], target_isotopes: dict[int, int]
    ) -> dict[str, Any]:
        import networkx as nx

        graph = nx.DiGraph()
        parent_id = "parent_0"
        parent_masses = [self.fetch_exact_mass(s) for s in base_molecule_symbols]
        graph.add_node(
            parent_id,
            symbols=base_molecule_symbols,
            total_mass=sum(parent_masses),
            is_parent=True,
        )
        branches: list[dict[str, Any]] = []
        nodes_dict: dict[str, Any] = {
            parent_id: {"symbols": base_molecule_symbols, "total_mass": sum(parent_masses)}
        }
        edges: list[dict[str, Any]] = []

        for idx, mass_num in target_isotopes.items():
            sym = base_molecule_symbols[idx]
            orig_mass = self.fetch_exact_mass(sym)
            iso_mass = self.fetch_exact_mass(sym, mass_num)
            delta_m = iso_mass - orig_mass
            child_id = f"iso_idx{idx}_{mass_num}{sym}"
            child_symbols = list(base_molecule_symbols)
            child_symbols[idx] = f"{mass_num}{sym}"
            graph.add_node(
                child_id,
                symbols=child_symbols,
                total_mass=sum(parent_masses) + delta_m,
                sub_idx=idx,
                delta_m=delta_m,
            )
            graph.add_edge(
                parent_id,
                child_id,
                sub_idx=idx,
                orig_symbol=sym,
                target_mass=mass_num,
                delta_m=delta_m,
            )
            branches.append(
                {
                    "branch_id": child_id,
                    "sub_idx": idx,
                    "original_symbol": sym,
                    "target_mass_num": mass_num,
                    "orig_mass": orig_mass,
                    "iso_mass": iso_mass,
                    "delta_m": delta_m,
                    "is_active": True,
                }
            )
            nodes_dict[child_id] = {
                "symbols": child_symbols,
                "total_mass": sum(parent_masses) + delta_m,
                "delta_m": delta_m,
            }
            edges.append({"parent": parent_id, "child": child_id, "sub_idx": idx, "delta_m": delta_m})

        return {"branches": branches, "nodes": nodes_dict, "edges": edges, "graph": graph}

    def fingerprint_payload(self, raw_data_dict: dict[str, Any]) -> str:
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        serialized = json.dumps(raw_data_dict, cls=_NumpyEncoder, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
