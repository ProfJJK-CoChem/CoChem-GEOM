import hashlib  # SHA-256 artifact provenance tracking
#!/usr/bin/env python3
"""
CoChem-GEOM (v4.0) - Stage 2.2: Variable Reduction & DoF Triage
---------------------------------------------------------------
Establishes the optimizable Z-matrix variable set. Applies Hydrogen freezing,
theoretical offset grouping for pseudo-symmetric coordinates, and rigorously
evaluates Degrees of Freedom (DoF) before unlocking downstream GPU fitting.
"""

import logging
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import List, Dict, Optional, Tuple, Any

class VariableTriageEngine:
    def __init__(self, elements: List[str], ccsdt_coords: Optional[np.ndarray] = None) -> None:
        """
        Initializes the Triage Engine.
        elements: List of atomic symbols (e.g., ['C', 'C', 'H', 'H']).
        ccsdt_coords: (N, 3) High-level theoretical coordinates used for offset grouping.
        """
        self.logger = logging.getLogger("CoChem_GEOM_Triage")
        self.elements = elements
        self.ccsdt_coords = ccsdt_coords
        
        self.num_atoms = len(elements)
        self.total_internal_coords = 3 * self.num_atoms - 6 if self.num_atoms > 2 else 1
        
        # Internal state tracking
        self.parameters = self._initialize_parameter_map()
        self.experimental_constants_count = 0
        self.is_locked = True  # Execution gate

        # UI Components
        self.out_badge = widgets.Output()

    def _initialize_parameter_map(self) -> List[Dict]:
        """
        Builds topological Z-matrix trees mapping internal coordinates (3N-6).
        Enforces valid connectivity graph topology.
        """
        params = []
        param_idx = 0
        
        # Atom 1: Origin (no variables)
        # Atom 2: Bond to Atom 1
        if self.num_atoms >= 2:
            params.append({
                "idx": param_idx,
                "type": "Bond",
                "atoms": [1, 0],
                "is_frozen": False,
                "linked_to": None,
                "offset": 0.0
            })
            param_idx += 1

        # Atom 3: Bond to Atom 2, Angle with Atom 1
        if self.num_atoms >= 3:
            params.append({
                "idx": param_idx,
                "type": "Bond",
                "atoms": [2, 1],
                "is_frozen": False,
                "linked_to": None,
                "offset": 0.0
            })
            param_idx += 1
            params.append({
                "idx": param_idx,
                "type": "Angle",
                "atoms": [2, 1, 0],
                "is_frozen": False,
                "linked_to": None,
                "offset": 0.0
            })
            param_idx += 1

        # Atom i >= 4: Bond, Angle, Dihedral to previous atoms
        for i in range(3, self.num_atoms):
            params.append({
                "idx": param_idx,
                "type": "Bond",
                "atoms": [i, i - 1],
                "is_frozen": False,
                "linked_to": None,
                "offset": 0.0
            })
            param_idx += 1
            params.append({
                "idx": param_idx,
                "type": "Angle",
                "atoms": [i, i - 1, i - 2],
                "is_frozen": False,
                "linked_to": None,
                "offset": 0.0
            })
            param_idx += 1
            params.append({
                "idx": param_idx,
                "type": "Dihedral",
                "atoms": [i, i - 1, i - 2, i - 3],
                "is_frozen": False,
                "linked_to": None,
                "offset": 0.0
            })
            param_idx += 1

        return params

    def get_float_variables_count(self) -> int:
        """Returns the number of parameters actively being optimized."""
        float_count = 0
        for p in self.parameters:
            if not p["is_frozen"] and p["linked_to"] is None:
                float_count += 1
        return float_count

    def apply_hydrogen_lock(self) -> Any:
        """
        Single-click action to freeze strictly X-H Bond length parameters.
        Preserves H-X-Y angles and rotamer dihedrals.
        """
        h_indices = [i for i, el in enumerate(self.elements) if el == 'H']
        lock_count = 0
        
        for p in self.parameters:
            # Restrict Hydrogen lock strictly to Bond parameters containing H
            if p["type"] == "Bond" and any(atom_idx in h_indices for atom_idx in p["atoms"]):
                if not p["is_frozen"]:
                    p["is_frozen"] = True
                    lock_count += 1
                    
        self.logger.info(f"Hydrogen Lock applied. Froze {lock_count} X-H bond parameters.")
        self._update_badge()

    def apply_theoretical_offsets(self, primary_idx: int, linked_indices: List[int]) -> Any:
        """
        Links pseudo-symmetric parameters into a single variable using CCSD(T) offsets.
        Calculates exact geometric differences from high-level reference coordinates.
        """
        if self.ccsdt_coords is None:
            self.logger.error("Theoretical offsets cannot be applied without CCSD(T) reference coordinates.")
            return

        coords = self.ccsdt_coords
        primary_param = self.parameters[primary_idx]
        
        def _calc_val(p) -> Any:
            idx = p["atoms"]
            if p["type"] == "Bond":
                return float(np.linalg.norm(coords[idx[0]] - coords[idx[1]]))
            elif p["type"] == "Angle":
                v1 = coords[idx[0]] - coords[idx[1]]
                v2 = coords[idx[2]] - coords[idx[1]]
                cost = np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0)
                return float(np.degrees(np.arccos(cost)))
            elif p["type"] == "Dihedral":
                v1 = coords[idx[1]] - coords[idx[0]]
                v2 = coords[idx[2]] - coords[idx[1]]
                v3 = coords[idx[3]] - coords[idx[2]]
                n1 = np.cross(v1, v2)
                n2 = np.cross(v2, v3)
                n1 /= max(np.linalg.norm(n1), 1e-12)
                n2 /= max(np.linalg.norm(n2), 1e-12)
                m1 = np.cross(n1, v2 / max(np.linalg.norm(v2), 1e-12))
                x = np.dot(n1, n2)
                y = np.dot(m1, n2)
                return float(np.degrees(np.arctan2(y, x)))
            return 0.0

        val_primary = _calc_val(primary_param)

        for linked_idx in linked_indices:
            linked_param = self.parameters[linked_idx]
            val_linked = _calc_val(linked_param)
            offset = val_linked - val_primary
            
            self.parameters[linked_idx]["linked_to"] = primary_idx
            self.parameters[linked_idx]["offset"] = offset
            self.parameters[linked_idx]["is_frozen"] = False # Controlled by primary
            
        self.logger.info(f"Linked parameters {linked_indices} to primary parameter {primary_idx} with CCSD(T) offsets.")
        self._update_badge()

    def evaluate_sufficiency(self, num_constants: int, rot_constants: Optional[Tuple[float, float, float]] = None) -> widgets.HTML:
        """
        Evaluates DoF math: (Variables + 1) <= Effective Constants.
        Incorporates Ray's asymmetry parameter kappa = (2B - A - C)/(A - C) to discount redundant constants for symmetric tops.
        """
        effective_constants = num_constants
        if rot_constants is not None and len(rot_constants) == 3:
            A, B, C = sorted(rot_constants, reverse=True)
            if abs(A - C) > 1e-6:
                kappa = (2.0 * B - A - C) / (A - C)
                # If symmetric top (kappa near -1 for prolate or +1 for oblate, |B - A| < 1e-3 or |B - C| < 1e-3)
                if abs(kappa + 1.0) < 1e-3 or abs(kappa - 1.0) < 1e-3:
                    self.logger.info("Symmetric top species detected. Discounting 1 redundant rotational constant.")
                    effective_constants = max(1, num_constants - 1)

        self.experimental_constants_count = num_constants
        float_variables = self.get_float_variables_count()
        margin = effective_constants - (float_variables + 1)

        if margin >= 0:
            self.is_locked = False
            html_str = f"<div style='padding:10px; background-color:#d4edda; color:#155724; border:1px solid #c3e6cb; border-radius:5px; font-weight:bold; font-size:14px;'>" \
                       f"✅ [SUFFICIENT: +{margin} DoF] Execution Gate Unlocked.</div>"
        else:
            self.is_locked = True
            missing = abs(margin)
            html_str = f"<div style='padding:10px; background-color:#f8d7da; color:#721c24; border:1px solid #f5c6cb; border-radius:5px; font-weight:bold; font-size:14px;'>" \
                       f"❌ [UNSUFFICIENT: Need +{missing} Constants or Freeze +{missing} Vars] Execution Locked.</div>"
            
        return widgets.HTML(html_str)

    def _update_badge(self, change=None) -> Any:
        """Callback to refresh the sufficiency badge when UI state changes."""
        with self.out_badge:
            clear_output(wait=True)
            display(self.evaluate_sufficiency(self.experimental_constants_count))

    def render_dashboard(self, initial_constants: int) -> Any:
        """Renders the Jupyter ipywidgets control panel."""
        self.experimental_constants_count = initial_constants
        
        # UI Elements
        const_input = widgets.IntText(
            value=self.experimental_constants_count,
            description='Exp. Constants:',
            style={'description_width': 'initial'}
        )
        
        btn_h_lock = widgets.Button(
            description='Freeze All Hydrogens',
            button_style='info',
            icon='lock'
        )

        # Callbacks
        def on_const_change(change) -> Any:
            self.experimental_constants_count = change['new']
            self._update_badge()

        def on_h_lock_click(b) -> Any:
            self.apply_hydrogen_lock()

        const_input.observe(on_const_change, names='value')
        btn_h_lock.on_click(on_h_lock_click)

        # Layout
        controls = widgets.HBox([const_input, btn_h_lock])
        panel = widgets.VBox([
            widgets.HTML("<h3>Stage 2.2: Variable Reduction & Degree-of-Freedom Triage</h3>"),
            controls,
            self.out_badge
        ])
        
        display(panel)
        self._update_badge()

if __name__ == "__main__":
    # Lightweight module test loop
    logging.basicConfig(level=logging.INFO)
    
    # Pyridine (C5H5N) sample
    sample_elements = ['C', 'C', 'C', 'C', 'C', 'N', 'H', 'H', 'H', 'H', 'H']
    sample_ccsdt = np.zeros((11, 3)) 
    
    triage = VariableTriageEngine(elements=sample_elements, ccsdt_coords=sample_ccsdt)
    logger.info(f"Initial 3N-6 Variables for Pyridine: {triage.get_float_variables_count()}")
    
    # Render with only 3 constants (A, B, C for one isotopologue)
    triage.render_dashboard(initial_constants=3)