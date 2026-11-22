#!/usr/bin/env python3
"""CoChem-GEOM Stage 4.0: LaTeX Report Generator.

Generates standalone and embeddable LaTeX tables comparing experimental and fitted
rotational constants, Kraitchman r_s coordinates (explicitly marking imaginary
axes without clipping), and separated r_e / r_s / r_0 / r_e^SE structural tables.
Contains ZERO hardcoded rotational constants.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from cochem_geom.reporting import render_structural_comparison_latex

logger = logging.getLogger(__name__)


def _escape_latex(text: str) -> str:
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&")


class GEOMReportLatexGenerator:
    """Generates LaTeX report tables and document sections for CoChem-GEOM fitting results."""

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path(os.environ.get("COCHEM_ARTIFACT_DIR", "."))

    def generate_rotational_constants_table(self, exp_constants: dict, fit_constants: dict) -> str:
        latex = [
            "\\begin{table}[h!]",
            "\\centering",
            "\\caption{Comparison of Experimental and Fitted Rotational Constants (MHz).}",
            "\\begin{tabular}{l c c c c}",
            "\\hline\\hline",
            "Species & Parameter & Experimental & Fitted & Residual (kHz) \\\\",
            "\\hline",
        ]

        for species_id, exp_dict in exp_constants.items():
            species_id_esc = _escape_latex(species_id)
            fit_dict = fit_constants.get(species_id, {})
            for param in ["A", "B", "C"]:
                if param in exp_dict:
                    exp_val_obj = exp_dict[param]
                    if isinstance(exp_val_obj, dict):
                        val_exp = float(exp_val_obj["value"])
                        err_exp = exp_val_obj.get("uncertainty")
                    else:
                        val_exp = float(exp_val_obj)
                        err_exp = None

                    fit_val_obj = fit_dict.get(param, val_exp)
                    val_fit = float(fit_val_obj["value"]) if isinstance(fit_val_obj, dict) else float(fit_val_obj)
                    residual_khz = (val_fit - val_exp) * 1000.0
                    err_str = f" $\\pm$ {float(err_exp):.4f}" if err_exp is not None else ""
                    latex.append(
                        f"{species_id_esc} & {param} & {val_exp:.4f}{err_str} & {val_fit:.4f} & {residual_khz:+.2f} \\\\"
                    )

        latex.extend(
            [
                "\\hline\\hline",
                "\\end{tabular}",
                "\\label{tab:rotational_constants}",
                "\\end{table}",
            ]
        )
        return "\n".join(latex)

    def generate_kraitchman_rs_table(self, rs_coordinates: dict) -> str:
        latex = [
            "\\begin{table}[h!]",
            "\\centering",
            "\\caption{Kraitchman Substitution Coordinates ($r_s$) in Principal Axes (\\AA).}",
            "\\begin{tabular}{l c c c}",
            "\\hline\\hline",
            "Atom & $|a|$ (\\AA) & $|b|$ (\\AA) & $|c|$ (\\AA) \\\\",
            "\\hline",
        ]

        for atom_name, coords in rs_coordinates.items():
            atom_name_esc = _escape_latex(atom_name)

            def format_coord(c: Any) -> str:
                if c is None:
                    return "imag"
                if np.iscomplexobj(c) or isinstance(c, complex):
                    if np.isnan(np.abs(c)):
                        return "---"
                    return f"{c.real:.4f}+{c.imag:.4f}i"
                if np.isnan(c):
                    return "---"
                return f"{np.abs(float(c)):.4f}"

            a_str = format_coord(coords[0])
            b_str = format_coord(coords[1])
            c_str = format_coord(coords[2])
            latex.append(f"{atom_name_esc} & {a_str} & {b_str} & {c_str} \\\\")

        latex.extend(
            [
                "\\hline\\hline",
                "\\end{tabular}",
                "\\label{tab:kraitchman_rs}",
                "\\end{table}",
            ]
        )
        return "\n".join(latex)

    def generate_structural_comparison_table(self, comparison_payload: dict[str, Any]) -> str:
        """Render separated r_e / r_s / r_0 / r_e^SE structural comparison table."""
        return render_structural_comparison_latex(comparison_payload)

    def generate_full_manuscript_section(
        self, species_name: str, rot_table: str, rs_table: str, symmetry: str = "C1"
    ) -> str:
        species_name_esc = _escape_latex(species_name)
        doc = [
            f"\\section{{Spectroscopic Structure Determination: {species_name_esc}}}",
            f"Spectroscopic fitting for \\textbf{{{species_name_esc}}} was executed under point-group symmetry \\texttt{{{symmetry}}}.",
            "",
            rot_table,
            "",
            rs_table,
            "",
        ]
        out_file = self.output_dir / f"{species_name}_geom_report.tex"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(doc)
        out_file.write_text(content, encoding="utf-8")
        return content
