#!/usr/bin/env python3
"""
CoChem-GEOM (v4.0) - Stage 2.1: Symmetry Detection & Thermal Warnings
---------------------------------------------------------------------
Determines point-group symmetry using molsym. Presents a Human-in-the-Loop 
(HITL) dashboard for manual override. Evaluates the Planar Inertial Defect 
to warn against thermal zero-point out-of-plane geometric distortions.
"""

import logging
import numpy as np
from typing import Optional, Dict, List, Any
import ipywidgets as widgets
from IPython.display import display, HTML
try:
    import molsym
    MOLSYM_AVAILABLE = True
except ImportError:
    molsym = None
    MOLSYM_AVAILABLE = False

class SymmetryControllerUI:
    def __init__(self) -> None:
        self.logger = logging.getLogger("CoChem_GEOM_Symmetry")
        self.symmetry_override_dict = {}

    def check_planar_inertial_defect(self, principal_moments: np.ndarray) -> Optional[widgets.HTML]:
        """
        Evaluates ΔI = Ic - Ia - Ib.
        principal_moments: [Ia, Ib, Ic] in amu*Å^2 (sorted smallest to largest)
        Dynamically scales inertial defect warning threshold based on total moment of inertia.
        """
        Ia, Ib, Ic = principal_moments
        delta_I = Ic - Ia - Ib
        total_I = np.sum(principal_moments)
        
        # Dynamically scale lower bound threshold for large polycyclics / PAHs
        lower_threshold = -max(1.5, 0.015 * total_I)
        
        if abs(delta_I) < 0.05:
            return widgets.HTML("<b style='color:green;'>[Planar]</b> Inertial Defect (ΔI) ≈ 0.")
        elif lower_threshold <= delta_I < -0.05:
            msg = f"<b style='color:orange;'>[Thermal Warning]</b> Negative Inertial Defect (ΔI = {delta_I:.3f} amu·Å²). " \
                  f"Geometry appears non-planar but is likely driven by out-of-plane zero-point vibrations. " \
                  f"Consider enforcing planarity constraints."
            self.logger.warning(f"Thermal Defect Detected: {delta_I:.3f}")
            return widgets.HTML(msg)
        elif delta_I < lower_threshold or delta_I > 0.05:
             return widgets.HTML(f"<b style='color:blue;'>[Non-Planar]</b> Inertial Defect (ΔI = {delta_I:.3f} amu·Å²).")
        return None

    def analyze_symmetry(self, coords: np.ndarray, elements: list, species_id: str) -> str:
        """
        Uses molsym to detect the point group.
        coords: (N,3) Cartesian coordinates.
        elements: List of atomic symbols (e.g., ['C', 'H', 'H']).
        Translates to Center of Mass and rounds to 1e-5 tolerance to prevent false C1 classification.
        """
        try:
            # Translate geometry to Center of Mass and round to 1e-5 tolerance
            com = np.mean(coords, axis=0)
            coords_com = coords - com
            coords_clean = np.round(coords_com / 1e-5) * 1e-5

            # Construct formatted string for molsym ingestion
            mol_str = f"{len(elements)}\n\n"
            for el, pt in zip(elements, coords_clean):
                mol_str += f"{el} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n"
                
            mol = molsym.Molecule.from_string(mol_str)
            pg = mol.point_group
            self.logger.info(f"[{species_id}] molsym detected Point Group: {pg}")
            return pg
        except Exception as e:
            self.logger.error(f"[{species_id}] molsym detection failed: {e}. Defaulting to C1.")
            return "C1"

    def render_hitl_dashboard(self, geometry_payloads: dict) -> Any:
        """
        Renders an interactive UI for all isomers/isotopologues.
        geometry_payloads: Dict containing coords, elements, and moments for each species.
        """
        ui_rows = []
        
        # Common Schönflies symbols for drop-down selection
        valid_pgs = ["C1", "Cs", "C2", "C2v", "C3v", "D2h", "D3h", "D6h", "Td", "Oh"]

        for species_id, data in geometry_payloads.items():
            # 1. Detect Base Symmetry
            detected_pg = self.analyze_symmetry(data['coords'], data['elements'], species_id)
            self.symmetry_override_dict[species_id] = detected_pg
            
            # 2. Check Inertial Defect
            defect_widget = self.check_planar_inertial_defect(data['principal_moments'])
            
            # 3. Build Dropdown
            # If molsym detected something weird, prepend it to the list
            opts = valid_pgs if detected_pg in valid_pgs else [detected_pg] + valid_pgs
            
            dropdown = widgets.Dropdown(
                options=opts,
                value=detected_pg,
                description=f"{species_id}:",
                style={'description_width': 'initial'},
                layout={'width': '250px'}
            )
            
            # Define callback for manual override
            def on_change(change, s_id=species_id) -> Any:
                if change['type'] == 'value':
                    self.logger.info(f"[{s_id}] Symmetry overridden by user: {change['old']} -> {change['new']}")
                    self.symmetry_override_dict[s_id] = change['new']
            
            dropdown.observe(on_change)
            
            # 4. Construct Row
            if defect_widget:
                row = widgets.HBox([dropdown, defect_widget])
            else:
                row = widgets.HBox([dropdown])
                
            ui_rows.append(row)
            
        # Display the final panel
        panel = widgets.VBox([
            widgets.HTML("<h3>Stage 2.1: Human-in-the-Loop Symmetry Override</h3>"),
            widgets.HTML("<i>Adjust point-groups if numerical grid noise reduced the detected symmetry.</i>"),
            *ui_rows
        ])
        display(panel)
        return panel

if __name__ == "__main__":
    # Lightweight module test loop (testing the Jupyter environment)
    logging.basicConfig(level=logging.INFO)
    sym_engine = SymmetryControllerUI()
    
    # Sample data payload for a nearly-planar molecule
    sample_payload = {
        "Iso_001_Water": {
            "coords": np.array([
                [ 0.000000,  0.000000,  0.117790],
                [ 0.000000,  0.755450, -0.471161],
                [ 0.000000, -0.755450, -0.471161]
            ]),
            "elements": ["O", "H", "H"],
            "principal_moments": np.array([0.5, 1.0, 1.4]) # Intentional small negative defect (1.4 - 1.5 = -0.1)
        }
    }
    
    # Render (will print HTML repr in terminal if ipywidgets is headless)
    sym_engine.render_hitl_dashboard(sample_payload)
    logger.info(f"Final Captured State: {sym_engine.symmetry_override_dict}")
