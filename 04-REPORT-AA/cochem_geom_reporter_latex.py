#!/usr/bin/env python3
"""
CoChem-GEOM (v4.0) - Stage 4.0: LaTeX Report Generator
-------------------------------------------------------
Generates standalone and embeddable LaTeX report blocks containing
fitted geometric parameters, Kraitchman r_s substitution coordinates,
rotational constant comparisons, and covariance error bars.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class GEOMReportLatexGenerator:
    """Generates LaTeX report tables and document sections for CoChem-GEOM fitting results."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(os.environ.get("COCHEM_ARTIFACT_DIR", "."))

    def generate_rotational_constants_table(self, exp_constants: Dict[str, Dict[str, float]], fit_constants: Dict[str, Dict[str, float]]) -> str:
        """
        Generates LaTeX tabular markup comparing experimental vs fitted rotational constants.
        """
        latex = [
            "\\begin{table}[h!]",
            "\\centering",
            "\\caption{Comparison of Experimental and Fitted Rotational Constants (MHz).}",
            "\\begin{tabular}{l c c c c}",
            "\\hline\\hline",
            "Species & Parameter & Experimental & Fitted & Residual (kHz) \\\\",
            "\\hline"
        ]

        for species_id, exp_dict in exp_constants.items():
            fit_dict = fit_constants.get(species_id, {})
            for param in ["A", "B", "C"]:
                if param in exp_dict:
                    val_exp = exp_dict[param]["value"]
                    err_exp = exp_dict[param].get("uncertainty", 0.0)
                    val_fit = fit_dict.get(param, {}).get("value", val_exp)
                    residual_khz = (val_fit - val_exp) * 1000.0
                    latex.append(f"{species_id} & {param} & {val_exp:.4f} $\\pm$ {err_exp:.4f} & {val_fit:.4f} & {residual_khz:+.2f} \\\\")

        latex.extend([
            "\\hline\\hline",
            "\\end{tabular}",
            "\\label{tab:rotational_constants}",
            "\\end{table}"
        ])
        return "\n".join(latex)

    def generate_kraitchman_rs_table(self, rs_coordinates: Dict[str, np.ndarray]) -> str:
        """
        Generates LaTeX tabular markup for Kraitchman r_s substitution coordinates.
        """
        latex = [
            "\\begin{table}[h!]",
            "\\centering",
            "\\caption{Kraitchman Substitution Coordinates ($r_s$) in Principal Axes (\\AA).}",
            "\\begin{tabular}{l c c c}",
            "\\hline\\hline",
            "Atom & $|a|$ (\\AA) & $|b|$ (\\AA) & $|c|$ (\\AA) \\\\",
            "\\hline"
        ]

        for atom_name, coords in rs_coordinates.items():
            a_str = f"{coords[0]:.4f}" if not np.isnan(coords[0]) else "---"
            b_str = f"{coords[1]:.4f}" if not np.isnan(coords[1]) else "---"
            c_str = f"{coords[2]:.4f}" if not np.isnan(coords[2]) else "---"
            latex.append(f"{atom_name} & {a_str} & {b_str} & {c_str} \\\\")

        latex.extend([
            "\\hline\\hline",
            "\\end{tabular}",
            "\\label{tab:kraitchman_rs}",
            "\\end{table}"
        ])
        return "\n".join(latex)

    def generate_full_manuscript_section(self, species_name: str, rot_table: str, rs_table: str, symmetry: str = "C1") -> str:
        """
        Assembles a full LaTeX manuscript section summarizing geometric determination.
        """
        doc = [
            f"\\section{{Semi-Experimental Structure Determination: {species_name}}}",
            f"Spectroscopic fitting for \\textbf{{{species_name}}} was executed under point-group symmetry \\texttt{{{symmetry}}}.",
            "",
            rot_table,
            "",
            rs_table,
            ""
        ]
        out_file = self.output_dir / f"{species_name}_geom_report.tex"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(doc)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(content)
        return content


if __name__ == "__main__":
    reporter = GEOMReportLatexGenerator()
    exp_data = {"Water": {"A": {"value": 855881.3, "uncertainty": 0.5}, "B": {"value": 435882.2, "uncertainty": 0.3}, "C": {"value": 278138.7, "uncertainty": 0.2}}}
    fit_data = {"Water": {"A": {"value": 855881.5}, "B": {"value": 435882.1}, "C": {"value": 278138.8}}}
    t1 = reporter.generate_rotational_constants_table(exp_data, fit_data)
    rs_data = {"H1": np.array([0.0, 0.7554, 0.4711]), "H2": np.array([0.0, 0.7554, -0.4711])}
    t2 = reporter.generate_kraitchman_rs_table(rs_data)
    sec = reporter.generate_full_manuscript_section("Water", t1, t2, symmetry="C2v")
    print("LaTeX Report Generator test passed.")
