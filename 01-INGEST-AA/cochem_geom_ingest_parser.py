#!/usr/bin/env python3
"""CoChem-GEOM Stage 1.0: Pickett (.var/.par/.fit/.lin) & CoChem-SPYFY JSON Ingestion.

Delegates to `cochem_geom.pickett` while preserving the `SpectraIngestionEngine`
and `ExperimentalConstant` compatibility API.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cochem_geom.pickett import (
    MalformedPickettError,
    PickettDeck,
    PickettFitSummary,
    PickettLine,
    determine_isotopic_abundance,
    parse_pickett_fit,
    parse_pickett_lin,
    parse_pickett_par,
    parse_pickett_var,
    parse_spyfy_json,
)

logger = logging.getLogger(__name__)


class ExperimentalConstant(BaseModel):
    species_id: str
    constant_type: str
    value_mhz: float
    uncertainty_mhz: float
    is_isotopologue: bool
    abundance_percentage: float
    source_file: str
    is_fitted_uncertainty: bool = True
    uncertainty_source: str = "var_fitted_std_err"


def _deck_to_constants(deck: PickettDeck) -> list[ExperimentalConstant]:
    return [
        ExperimentalConstant(
            species_id=p.species_id,
            constant_type=p.name,
            value_mhz=p.value_mhz,
            uncertainty_mhz=p.uncertainty_mhz,
            is_isotopologue=p.is_isotopologue,
            abundance_percentage=p.abundance_percentage,
            source_file=p.source_file,
            is_fitted_uncertainty=p.is_fitted_uncertainty,
            uncertainty_source=p.uncertainty_source,
        )
        for p in deck.parameters
    ]


class SpectraIngestionEngine:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.logger = logging.getLogger("CoChem_GEOM_Ingest")
        self.triage_log: list[dict[str, Any]] = []

    def _determine_abundance(self, species_id: str) -> float:
        return determine_isotopic_abundance(species_id)

    def parse_pickett_var(self, var_path: Path) -> PickettDeck:
        return parse_pickett_var(var_path)

    def parse_pickett_par(
        self, par_path: Path, *, allow_negative_step_abs: bool = True
    ) -> list[ExperimentalConstant]:
        deck = parse_pickett_par(par_path, allow_negative_step_abs=allow_negative_step_abs)
        return _deck_to_constants(deck)

    def parse_pickett_lin(self, lin_path: Path) -> list[PickettLine]:
        return parse_pickett_lin(lin_path)

    def parse_pickett_fit(self, fit_path: Path) -> PickettFitSummary:
        return parse_pickett_fit(fit_path)

    def parse_spyfy_json(self, json_path: Path) -> list[ExperimentalConstant]:
        deck = parse_spyfy_json(json_path)
        return _deck_to_constants(deck)

    def filter_isotopic_abundance(
        self, constants: list[ExperimentalConstant], threshold: float = 0.01
    ) -> list[ExperimentalConstant]:
        valid_constants: list[ExperimentalConstant] = []
        for c in constants:
            if c.abundance_percentage < threshold:
                self.triage_log.append(
                    {
                        "species": c.species_id,
                        "issue": "suppressed_due_to_noise",
                        "abundance": c.abundance_percentage,
                        "threshold_required": threshold,
                    }
                )
            else:
                valid_constants.append(c)
        return valid_constants

    def write_triage_report(self) -> Path:
        report_path = self.workspace_dir / "reports" / "triage_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.triage_log, indent=4), encoding="utf-8")
        return report_path


__all__ = [
    "ExperimentalConstant",
    "MalformedPickettError",
    "SpectraIngestionEngine",
]
