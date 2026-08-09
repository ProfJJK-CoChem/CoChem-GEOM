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
from typing import List, Dict, Optional

class VariableTriageEngine:
    def __init__(self, elements: List[str], ccsdt_coords: Optional[np.ndarray] = None):
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
        Builds the baseline metadata matrix for internal coordinates (3N-6).
        Actual Z-Matrix mapping occurs in Stage 3.1; this maps the variables.
        """
        params = []
        # Mock parameter generation representing Bonds, Angles, Dihedrals
        for i in range(self.total_internal_coords):
            # Simplistic assignment for triage demonstration
            p_type = "Bond" if i < (self.num_atoms - 1) else ("Angle" if i < (2 * self.num_atoms - 3) else "Dihedral")
            
            # Map mock atom indices based on type
            if p_type == "Bond":
                atoms = [i, i + 1]
            elif p_type == "Angle":
                atoms = [max(0, i - self.num_atoms), i, i + 1]
            else:
                atoms = [max(0, i - 2*self.num_atoms), max(0, i - self.num_atoms), i, i + 1]

            params.append({
                "idx": i,
                "type": p_type,
                "atoms": atoms,
                "is_frozen": False,
                "linked_to": None,
                "offset": 0.0
            })
        return params

    def get_float_variables_count(self) -> int:
        """Returns the number of parameters actively being optimized."""
        float_count = 0
        for p in self.parameters:
            if not p["is_frozen"] and p["linked_to"] is None:
                float_count += 1
        return float_count

    def apply_hydrogen_lock(self):
        """
        Single-click action to freeze all parameters involving Hydrogen atoms.
        Drastically reduces DoF for complex organic structures.
        """
        h_indices = [i for i, el in enumerate(self.elements) if el == 'H']
        lock_count = 0
        
        for p in self.parameters:
            # If any atom in this coordinate is a Hydrogen, freeze it
            if any(atom_idx in h_indices for atom_idx in p["atoms"]):
                if not p["is_frozen"]:
                    p["is_frozen"] = True
                    lock_count += 1
                    
        self.logger.info(f"Hydrogen Lock applied. Froze {lock_count} parameters.")
        self._update_badge()

    def apply_theoretical_offsets(self, primary_idx: int, linked_indices: List[int]):
        """
        Links pseudo-symmetric parameters into a single variable using CCSD(T) offsets.
        """
        if self.ccsdt_coords is None:
            self.logger.error("Theoretical offsets cannot be applied without CCSD(T) reference coordinates.")
            return

        for linked_idx in linked_indices:
            # In a full implementation, distance/angle differences are calculated here 
            # from self.ccsdt_coords based on the Z-matrix topology.
            mock_offset = 0.005 # Mock physical offset in Ångströms/Degrees
            
            self.parameters[linked_idx]["linked_to"] = primary_idx
            self.parameters[linked_idx]["offset"] = mock_offset
            self.parameters[linked_idx]["is_frozen"] = False # Controlled by primary
            
        self.logger.info(f"Linked parameters {linked_indices} to primary parameter {primary_idx}.")
        self._update_badge()

    def evaluate_sufficiency(self, num_constants: int) -> widgets.HTML:
        """
        Evaluates DoF math: (Variables + 1) <= Constants.
        Generates the strict execution gate badge.
        """
        self.experimental_constants_count = num_constants
        float_variables = self.get_float_variables_count()
        margin = num_constants - (float_variables + 1)

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

    def _update_badge(self, change=None):
        """Callback to refresh the sufficiency badge when UI state changes."""
        with self.out_badge:
            clear_output(wait=True)
            display(self.evaluate_sufficiency(self.experimental_constants_count))

    def render_dashboard(self, initial_constants: int):
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
        def on_const_change(change):
            self.experimental_constants_count = change['new']
            self._update_badge()

        def on_h_lock_click(b):
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
    
    # Mocking Pyridine (C5H5N)
    mock_elements = ['C', 'C', 'C', 'C', 'C', 'N', 'H', 'H', 'H', 'H', 'H']
    mock_ccsdt = np.zeros((11, 3)) 
    
    triage = VariableTriageEngine(elements=mock_elements, ccsdt_coords=mock_ccsdt)
    print(f"Initial 3N-6 Variables for Pyridine: {triage.get_float_variables_count()}")
    
    # Render with only 3 constants (A, B, C for one isotopologue)
    triage.render_dashboard(initial_constants=3)