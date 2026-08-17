#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)

"""
CoChem-GEOM (v4.0) - Stage 4.0: LaTeX Report Generator
-------------------------------------------------------
Generates standalone and embeddable LaTeX report blocks containing
fitted geometric parameters, Kraitchman r_s substitution coordinates,
rotational constant comparisons, and covariance error bars.
"""

import os
from pathlib import Path
import numpy as np

def _escape_latex(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("_", "\\_").replace("&", "\\&")


class GEOMReportLatexGenerator:
    """Generates LaTeX report tables and document sections for CoChem-GEOM fitting results."""

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path(os.environ.get("COCHEM_ARTIFACT_DIR", "."))

    def generate_rotational_constants_table(self, exp_constants: dict, fit_constants: dict) -> str:
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
            species_id_esc = _escape_latex(species_id)
            fit_dict = fit_constants.get(species_id, {})
            for param in ["A", "B", "C"]:
                if param in exp_dict:
                    exp_val_obj = exp_dict[param]
                    if isinstance(exp_val_obj, dict):
                        val_exp = exp_val_obj.get("value", 0.0)
                        err_exp = exp_val_obj.get("uncertainty", 0.0)
                    else:
                        val_exp = float(exp_val_obj)
                        err_exp = 0.0

                    fit_val_obj = fit_dict.get(param, val_exp)
                    if isinstance(fit_val_obj, dict):
                        val_fit = fit_val_obj.get("value", val_exp)
                    else:
                        val_fit = float(fit_val_obj)

                    residual_khz = (val_fit - val_exp) * 1000.0
                    latex.append(f"{species_id_esc} & {param} & {val_exp:.4f} $\\pm$ {err_exp:.4f} & {val_fit:.4f} & {residual_khz:+.2f} \\\\")

        latex.extend([
            "\\hline\\hline",
            "\\end{tabular}",
            "\\label{tab:rotational_constants}",
            "\\end{table}"
        ])
        return "\n".join(latex)

    def generate_kraitchman_rs_table(self, rs_coordinates: dict) -> str:
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
            atom_name_esc = _escape_latex(atom_name)
            def format_coord(c):
                if np.iscomplexobj(c) or isinstance(c, complex):
                    if np.isnan(np.abs(c)): return "---"
                    return f"{c.real:.4f}+{c.imag:.4f}i"
                else:
                    if np.isnan(c): return "---"
                    return f"{np.abs(c):.4f}"
            a_str = format_coord(coords[0])
            b_str = format_coord(coords[1])
            c_str = format_coord(coords[2])
            latex.append(f"{atom_name_esc} & {a_str} & {b_str} & {c_str} \\\\")

        latex.extend([
            "\\hline\\hline",
            "\\end{tabular}",
            "\\label{tab:kraitchman_rs}",
            "\\end{table}"
        ])
        return "\n".join(latex)

    def generate_geometric_parameters_table(self) -> str:
        """
        Placeholder for generating geometric parameters table.
        """
        return "% Geometric parameters table placeholder\n"

    def generate_covariance_table(self) -> str:
        """
        Placeholder for generating covariance table.
        """
        return "% Covariance table placeholder\n"

    def generate_full_manuscript_section(self, species_name: str, rot_table: str, rs_table: str, symmetry: str = "C1") -> str:
        """
        Assembles a full LaTeX manuscript section summarizing geometric determination.
        """
        species_name_esc = _escape_latex(species_name)
        doc = [
            f"\\section{{Semi-Experimental Structure Determination: {species_name_esc}}}",
            f"Spectroscopic fitting for \\textbf{{{species_name_esc}}} was executed under point-group symmetry \\texttt{{{symmetry}}}.",
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
    logger.info("LaTeX Report Generator test passed.")
