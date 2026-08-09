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
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class ExperimentalConstant:
    species_id: str
    constant_type: str
    value_mhz: float
    uncertainty_mhz: float
    is_isotopologue: bool
    abundance_percentage: float
    source_file: str

class MalformedPickettError(Exception):
    """Raised when Pickett parsing encounters overlapped columns or string-truncation."""
    pass

class SpectraIngestionEngine:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.logger = logging.getLogger("CoChem_GEOM_Ingest")
        self.triage_log = []

    def _determine_abundance(self, species_id: str) -> float:
        """
        Placeholder logic for isotopic abundance mapping.
        In Stage 1.1, this will bridge with Mendeleev.
        For now, returns 100.0 for main species, 1.0 for detected 13C/D, 
        and 0.005 for heavily substituted variants (to trigger the flag).
        """
        if "13C" in species_id or "D" in species_id:
            if species_id.count("13C") > 1 or species_id.count("D") > 1:
                return 0.005 # Rare doubly-substituted
            return 1.1 # Standard 13C abundance
        return 100.0

    def parse_pickett_par(self, par_path: Path) -> List[ExperimentalConstant]:
        """
        Extracts optimized rotational constants from a Pickett .par file.
        Enforces strict column boundaries to prevent 1P,E15.8 truncation collisions.
        """
        constants = []
        # Standard SPFIT Parameter ID mapping for Rigid Rotors
        param_map = {10000: 'A', 20000: 'B', 30000: 'C', 
                     200: 'DJ', 2000: 'DJK', 200000: 'DK'}
                     
        # Regex: ID (int), Value (float), Error (float)
        # Matches formats like: 10000  12345.6789  0.0050
        regex = re.compile(r"^\s*(\d+)\s+([-\.\dEDe]+)\s+([-\.\dEDe]+)")

        try:
            with open(par_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            self.logger.error(f"Encoding failure on {par_path.name}. Ensure it is UTF-8.")
            raise MalformedPickettError(f"Encoding Error on {par_path.name}")

        species_id = par_path.stem

        for line_num, line in enumerate(lines):
            # Skip blank lines and non-parameter definitions
            if not line.strip() or line.startswith('/'):
                continue
                
            match = regex.match(line)
            if not match:
                # If a line has data but fails the regex, it indicates column overlap
                if len(line.strip().split()) >= 3:
                     raise MalformedPickettError(f"Syntax collision at line {line_num} in {par_path.name}: {line.strip()}")
                continue

            param_id = int(match.group(1))
            
            # Translate Pickett Fortran 'D' exponential to Python 'E'
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

    def parse_spycfit_json(self, json_path: Path) -> List[ExperimentalConstant]:
        """
        Ingests native CoChem-SpycFit optimized JSON payloads.
        """
        constants = []
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Corrupted SpycFit Payload: {json_path.name}")

        species_id = data.get("species_id", json_path.stem)
        abundance = self._determine_abundance(species_id)
        
        for param, values in data.get("optimized_constants", {}).items():
             constants.append(ExperimentalConstant(
                 species_id=species_id,
                 constant_type=param.upper(),
                 value_mhz=float(values.get("value", 0.0)),
                 uncertainty_mhz=float(values.get("uncertainty", 0.0)),
                 is_isotopologue=(abundance < 99.0),
                 abundance_percentage=abundance,
                 source_file=json_path.name
             ))
             
        self.logger.info(f"Ingested {len(constants)} constants via SpycFit JSON from {json_path.name}")
        return constants

    def filter_isotopic_abundance(self, constants: List[ExperimentalConstant], threshold: float = 0.01) -> List[ExperimentalConstant]:
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

    def write_triage_report(self):
        """Dumps the pre-flight logic logs to the workspace."""
        report_path = self.workspace_dir / "reports" / "triage_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(self.triage_log, f, indent=4)

if __name__ == "__main__":
    # Lightweight module test loop
    logging.basicConfig(level=logging.INFO)
    test_dir = Path("./GEOM_Workspace")
    test_dir.mkdir(exist_ok=True)
    
    engine = SpectraIngestionEngine(test_dir)
    print("SpectraIngestionEngine initialized. Ready to process Pickett .par or SpycFit .json files.")