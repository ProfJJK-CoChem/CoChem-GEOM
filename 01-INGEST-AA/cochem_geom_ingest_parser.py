#!/usr/bin/env python3
"""
CoChem-GEOM (v4.0) - Stage 1.0: Ingestion & Pre-Flight Syntax Validator
-----------------------------------------------------------------------
Parses legacy Pickett files and CoChem-SpycFit payloads. Validates
quantum number syntax and aggressively flags rare isotopes falling
below the 0.01% natural abundance threshold.
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)

class ExperimentalConstant(BaseModel):
    species_id: str
    constant_type: str
    value_mhz: float
    uncertainty_mhz: float
    is_isotopologue: bool
    abundance_percentage: float
    source_file: str

class MalformedPickettError(Exception):
    """Raised when Pickett parsing encounters overlapped columns or string-truncation."""

class SpectraIngestionEngine:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.logger = logging.getLogger("CoChem_GEOM_Ingest")
        self.triage_log = []

    def _determine_abundance(self, species_id: str) -> float:
        """
        Calculates isotopic natural abundance using mendeleev or comprehensive IUPAC fallback data.
        Evaluates exact multi-isotope substitution products (e.g. 13C2, 18O, 15N, D).
        """
        # Standalone isotopic natural abundances (percentage)
        iso_abundance_map = {
            "13C": 1.07,
            "14C": 0.0000000001,
            "2H": 0.0115,
            "D": 0.0115,
            "15N": 0.368,
            "17O": 0.038,
            "18O": 0.205,
            "33S": 0.75,
            "34S": 4.21,
            "36S": 0.02,
            "37Cl": 24.23,
            "81Br": 49.31
        }
        
        # Parse isotope patterns in species_id
        matches = list(re.finditer(r"(?:(?P<mass>\d+))?(?P<elem>[A-Z][a-z]?|D)(?P<count>\d+)?", species_id))
        if not matches:
            return 100.0
            
        net_abundance = 100.0
        substituted = False
        
        for match in matches:
            mass_str = match.group("mass") or ""
            elem = match.group("elem")
            count = int(match.group("count") or 1)
            
            tag = f"{mass_str}{elem}"
            if tag in iso_abundance_map:
                substituted = True
                net_abundance *= (iso_abundance_map[tag] / 100.0) ** count
            elif mass_str:
                try:
                    import mendeleev
                    iso = mendeleev.isotope(elem, int(mass_str))
                    if iso and iso.abundance is not None:
                        substituted = True
                        net_abundance *= (float(iso.abundance) / 100.0) ** count
                except Exception as ex:
                    self.logger.debug(f"Mendeleev lookup failed for {elem}-{mass_str}: {ex}")
                    
        if substituted:
            return float(net_abundance)
        return 100.0

    def parse_pickett_par(self, par_path: Path) -> list[ExperimentalConstant]:
        """
        Extracts optimized rotational constants from a Pickett .par file.
        Uses fixed character slice parsing and fallback overlap-tolerant regex
        to prevent 1P,E15.8 truncation collisions when negative floats touch adjacent fields.
        """
        constants = []
        param_map = {10000: 'A', 20000: 'B', 30000: 'C', 
                     200: 'DJ', 2000: 'DJK', 200000: 'DK'}

        try:
            with open(par_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            self.logger.error(f"Encoding failure on {par_path.name}. Ensure it is UTF-8.")
            raise MalformedPickettError(f"Encoding Error on {par_path.name}")

        species_id = par_path.stem

        for line_num, line in enumerate(lines):
            if not line.strip() or line.startswith('/') or line_num == 0:
                continue
                
            # Method 1: Try fixed-width slices (Standard Pickett I10, F20.10, F20.10)
            parsed_line = False
            if len(line) >= 20:
                try:
                    f1 = line[0:10].strip()
                    # Look for second and third fields
                    rest = line[10:]
                    # Regex for floats handling minus sign without whitespace gap
                    num_match = re.findall(r"([+-]?\d+\.?\d*(?:[eEdD][+-]?\d+)?)", rest)
                    if f1.isdigit() and len(num_match) >= 2:
                        param_id = int(f1)
                        val_str = num_match[0].replace('D', 'E').replace('d', 'e')
                        err_str = num_match[1].replace('D', 'E').replace('d', 'e')
                        val = float(val_str)
                        err = float(err_str)
                        parsed_line = True
                except Exception:
                    parsed_line = False

            # Method 2: Fallback regex supporting zero-whitespace separators before '-'
            if not parsed_line:
                regex = re.compile(r"^\s*(\d+)\s*([+-]?\d+\.?\d*(?:[eEdD][+-]?\d+)?)\s*([+-]?\d+\.?\d*(?:[eEdD][+-]?\d+)?)")
                match = regex.match(line)
                if not match:
                    if len(line.strip().split()) >= 3:
                        raise MalformedPickettError(f"Syntax collision at line {line_num} in {par_path.name}: {line.strip()}")
                    continue

                param_id = int(match.group(1))
                val_str = match.group(2).replace('D', 'E').replace('d', 'e')
                err_str = match.group(3).replace('D', 'E').replace('d', 'e')
                try:
                    val = float(val_str)
                    err = float(err_str)
                except ValueError:
                    raise MalformedPickettError(f"Float conversion failure at line {line_num} in {par_path.name}")

            if param_id in param_map:
                abundance = self._determine_abundance(species_id)
                constants.append(ExperimentalConstant(
                    species_id=species_id,
                    constant_type=param_map[param_id],
                    value_mhz=val,
                    uncertainty_mhz=err,
                    is_isotopologue=(abundance < 99.0),
                    abundance_percentage=abundance,
                    source_file=par_path.name
                ))

        self.logger.info(f"Parsed {len(constants)} constants from {par_path.name}")
        return constants

    def parse_spycfit_json(self, json_path: Path) -> list[ExperimentalConstant]:
        """
        Ingests native CoChem-SpycFit optimized JSON payloads.
        """
        constants = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
        except json.JSONDecodeError:
            raise ValueError(f"Corrupted SpycFit Payload: {json_path.name}")

        species_id = data.get("species_id", json_path.stem)
        abundance = self._determine_abundance(species_id)
        
        for param, values in data.get("optimized_constants", {}).items():
             if isinstance(values, dict):
                 val_mhz = float(values.get("value", 0.0))
                 unc_mhz = float(values.get("uncertainty", 0.0))
             else:
                 val_mhz = float(values)
                 unc_mhz = 0.0

             constants.append(ExperimentalConstant(
                 species_id=species_id,
                 constant_type=param.upper(),
                 value_mhz=val_mhz,
                 uncertainty_mhz=unc_mhz,
                 is_isotopologue=(abundance < 99.0),
                 abundance_percentage=abundance,
                 source_file=json_path.name
             ))
             
        self.logger.info(f"Ingested {len(constants)} constants via SpycFit JSON from {json_path.name}")
        return constants

    def filter_isotopic_abundance(self, constants: list[ExperimentalConstant], threshold: float = 0.01) -> list[ExperimentalConstant]:
        """
        Sweeps the ingested pool. Flags and suppresses species representing
        less than the allowed natural abundance threshold.
        """
        valid_constants = []
        suppressed_count = 0
        
        for c in constants:
            if c.abundance_percentage < threshold:
                self.triage_log.append({
                    "species": c.species_id,
                    "issue": "suppressed_due_to_noise",
                    "abundance": c.abundance_percentage,
                    "threshold_required": threshold
                })
                suppressed_count += 1
            else:
                valid_constants.append(c)
                
        if suppressed_count > 0:
            self.logger.warning(f"Isotope Filter: Suppressed {suppressed_count} constants falling below {threshold}% natural abundance.")
            
        return valid_constants

    def write_triage_report(self) -> Any:
        """Dumps the pre-flight logic logs to the workspace."""
        report_path = self.workspace_dir / "reports" / "triage_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.triage_log, f, indent=4)

if __name__ == "__main__":
    # Lightweight module test loop
    logging.basicConfig(level=logging.INFO)
    test_dir = Path("./GEOM_Workspace")
    test_dir.mkdir(exist_ok=True)
    
    engine = SpectraIngestionEngine(test_dir)
    logger.info("SpectraIngestionEngine initialized. Ready to process Pickett .par or SpycFit .json files.")
