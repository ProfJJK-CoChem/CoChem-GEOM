#!/usr/bin/env python3
"""
CoChem-GEOM (v4.0) - Stage 2.1: Symmetry Detection & Thermal Warnings
---------------------------------------------------------------------
Determines point-group symmetry using molsym. Presents a Human-in-the-Loop 
(HITL) dashboard for manual override. Evaluates the Planar Inertial Defect 
to warn against thermal zero-point out-of-plane geometric distortions.
"""

import html
import logging
import warnings
import numpy as np
from typing import Any
import ipywidgets as widgets
from IPython.display import display, HTML
try:
    import molsym
    MOLSYM_AVAILABLE = True
except ImportError:
    molsym = None
    MOLSYM_AVAILABLE = False

class SymmetryControllerUI:
    def __init__(self, *, use_molsym: bool = True) -> None:
        self.logger = logging.getLogger("CoChem_GEOM_Symmetry")
        self.symmetry_override_dict: dict[str, str] = {}
        self.symmetry_warnings: dict[str, str] = {}
        self.last_symmetry_warning: str | None = None
        self.use_molsym = use_molsym

    def _record_c1_fallback(self, species_id: str, reason: str, *, is_error: bool = False) -> str:
        """Record user-visible fallback state and emit UserWarning when defaulting to C1."""
        warning_msg = (
            f"[{species_id}] {reason}. Falling back to unverified point group 'C1'; "
            "manual symmetry verification/override is required."
        )
        if is_error:
            self.logger.error(warning_msg)
        else:
            self.logger.warning(warning_msg)
        self.last_symmetry_warning = warning_msg
        self.symmetry_warnings[species_id] = warning_msg
        warnings.warn(warning_msg, UserWarning, stacklevel=3)
        return "C1"

    def check_planar_inertial_defect(self, principal_moments: np.ndarray) -> widgets.HTML | None:
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
        
        if delta_I > 0.2 or delta_I < lower_threshold:
             return widgets.HTML(f"<b style='color:blue;'>[Non-Planar]</b> Inertial Defect (ΔI = {delta_I:.3f} amu·Å²).")
        elif lower_threshold <= delta_I <= -0.05:
            msg = f"<b style='color:orange;'>[Thermal Warning]</b> Negative Inertial Defect (ΔI = {delta_I:.3f} amu·Å²). " \
                  f"Geometry appears non-planar but is likely driven by out-of-plane zero-point vibrations. " \
                  f"Consider enforcing planarity constraints."
            self.logger.warning(f"Thermal Defect Detected: {delta_I:.3f}")
            return widgets.HTML(msg)
        else:
             return widgets.HTML(f"<b style='color:green;'>[Planar]</b> Inertial Defect (ΔI = {delta_I:.3f} amu·Å²) is within valid planar range.")

    def analyze_symmetry(self, coords: np.ndarray, elements: list[str], species_id: str) -> str:
        """
        Uses molsym to detect the point group.
        coords: (N,3) Cartesian coordinates.
        elements: List of atomic symbols (e.g., ['C', 'H', 'H']).
        Translates to Geometric Centroid and rounds to 1e-5 tolerance to prevent false C1 classification.
        Emits UserWarning and records `last_symmetry_warning` / `symmetry_warnings[species_id]`
        whenever molsym is unavailable or fails and falls back to unverified `"C1"`.
        """
        if not self.use_molsym or not MOLSYM_AVAILABLE or molsym is None:
            return self._record_c1_fallback(
                species_id,
                "molsym not available",
                is_error=False,
            )

        try:
            coords_arr = np.asarray(coords, dtype=np.float64)
            if coords_arr.ndim != 2 or coords_arr.shape[1] != 3 or coords_arr.shape[0] == 0:
                raise ValueError(f"Expected non-empty (N, 3) coordinate array, got shape {coords_arr.shape}")
            if len(elements) != coords_arr.shape[0]:
                raise ValueError(
                    f"Element count ({len(elements)}) does not match coordinate rows ({coords_arr.shape[0]})"
                )
            if not np.all(np.isfinite(coords_arr)):
                raise ValueError("Cartesian coordinates contain non-finite values")

            # Translate geometry to Geometric Centroid and round to 1e-5 tolerance
            centroid = np.mean(coords_arr, axis=0)
            coords_centered = coords_arr - centroid
            coords_clean = np.round(coords_centered / 1e-5) * 1e-5

            if hasattr(molsym.Molecule, "from_string"):
                mol_str = f"{len(elements)}\n\n"
                for el, pt in zip(elements, coords_clean):
                    mol_str += f"{el} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n"
                mol = molsym.Molecule.from_string(mol_str)
                pg = str(mol.point_group)
            else:
                mol = molsym.Molecule.from_schema(
                    {
                        "symbols": list(elements),
                        "geometry": coords_clean.reshape(-1).tolist(),
                    }
                )
                pg = str(molsym.find_point_group(mol)[0])

            self.last_symmetry_warning = None
            self.symmetry_warnings.pop(species_id, None)
            self.logger.info(f"[{species_id}] molsym detected Point Group: {pg}")
            return pg
        except Exception as e:
            return self._record_c1_fallback(
                species_id,
                f"molsym detection failed: {e}",
                is_error=True,
            )

    def render_hitl_dashboard(self, geometry_payloads: dict[str, Any]) -> Any:
        """
        Renders an interactive UI for all isomers/isotopologues.
        geometry_payloads: Dict containing coords, elements, and moments for each species.
        Displays an explicit warning banner whenever any species falls back to unverified C1.
        """
        ui_rows = []
        fallback_notes: list[str] = []

        # Common Schönflies symbols for drop-down selection
        valid_pgs = ["C1", "Cs", "C2", "C2v", "C3v", "D2h", "D3h", "D6h", "Td", "Oh"]

        for species_id, data in geometry_payloads.items():
            # 1. Detect Base Symmetry
            detected_pg = self.analyze_symmetry(data['coords'], data['elements'], species_id)
            self.symmetry_override_dict[species_id] = detected_pg
            fallback_warning = self.symmetry_warnings.get(species_id)

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
            def on_change(change: dict[str, Any], s_id: str = species_id) -> None:
                self.logger.info(f"[{s_id}] Symmetry overridden by user: {change['old']} -> {change['new']}")
                self.symmetry_override_dict[s_id] = change['new']

            dropdown.observe(on_change, names='value')

            # 4. Construct Row
            row_children: list[Any] = [dropdown]
            if defect_widget:
                row_children.append(defect_widget)
            if fallback_warning:
                fallback_notes.append(fallback_warning)
                row_children.append(
                    widgets.HTML(
                        "<span style='color:#b45309; background:#fef3c7; border:1px solid #f59e0b; "
                        f"padding:2px 6px;'><b>[Unverified C1 Fallback]</b> {html.escape(fallback_warning)}</span>"
                    )
                )
            ui_rows.append(widgets.HBox(row_children))

        panel_children: list[Any] = [
            widgets.HTML("<h3>Stage 2.1: Human-in-the-Loop Symmetry Override</h3>"),
            widgets.HTML("<i>Adjust point-groups if numerical grid noise reduced the detected symmetry.</i>"),
        ]
        if fallback_notes:
            banner_items = "".join(f"<li>{html.escape(note)}</li>" for note in fallback_notes)
            panel_children.append(
                widgets.HTML(
                    "<div style='color:#92400e; background:#fef3c7; border:1px solid #f59e0b; "
                    "padding:8px 12px; margin:6px 0;'>"
                    "<b>[Symmetry Fallback Warning — Unverified C1]</b> "
                    "Automatic symmetry detection fell back to unverified <code>C1</code> for one or more species:"
                    f"<ul>{banner_items}</ul></div>"
                )
            )
        panel_children.extend(ui_rows)

        # Display the final panel
        panel = widgets.VBox(panel_children)
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
    logging.info(f"Final Captured State: {sym_engine.symmetry_override_dict}")
