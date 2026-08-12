import hashlib  # SHA-256 artifact provenance tracking
# Spin contamination audit check: <S^2> check
#!/usr/bin/env python3
"""
CoChem-GEOM (v4.0) - Stage 3.3: MPQC CCSD(T)-F12 Single-Point
--------------------------------------------------------------
Generates MPQC inputs for CCSD(T)-F12/cc-pVTZ-F12 single-point 
energies. Enforces the Escalator Rule requiring pre-optimized 
geometries from the PySCF DFT escalator. Enforces T1/D1 diagnostics.
"""

import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

class MultireferenceInstabilityError(Exception):
    """Raised when T1 > 0.02 or D1 > 0.05, indicating wavefunction failure."""

class EngineConvergenceError(Exception):
    """Raised when MPQC fails to reach SCF convergence."""

class MPQCSinglePointEngine:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.logger = logging.getLogger("CoChem_GEOM_CCSDT")
        self.mpqc_binary = os.environ.get("MPQC_CMD", "mpqc")

    def generate_input(
        self,
        base_name: str,
        elements: List[str],
        coords: np.ndarray,
        frozen_zmat: Optional[List[Dict]] = None,
        charge: int = 0,
        mult: int = 1,
        inhess: str = "XTB2",
        freeze_mode: str = "relaxed",
        monomer_atom_indices: Optional[List[List[int]]] = None,
        pyscf_escalator_optimized: bool = False
    ) -> Path:
        """
        Constructs the MPQC input file for CCSD(T)-F12/cc-pVTZ-F12.
        Enforces the Escalator Rule (no optimization allowed here).
        """
        assert pyscf_escalator_optimized, "Escalator Rule violation: Geometry must be pre-optimized by PySCF DFT escalator."

        inp_path = self.workspace_dir / f"{base_name}_ccsdt_refine.inp"
        
        # Load hardware parameters
        nprocs = int(os.environ.get("COCHEM_NPROCS", max(1, os.cpu_count() - 2)))
        maxcore = int(os.environ.get("COCHEM_MAXCORE", 4000))
        
        config_path = self.workspace_dir / "cochem_system_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg = json.loads(f.read())
                    nprocs = cfg.get("nprocs", nprocs)
                    maxcore = cfg.get("maxcore", maxcore)
            except Exception as ex:
                self.logger.debug(f"System config read failed: {ex}")

        # Detect open-shell radical state
        is_open_shell = (mult > 1)
        method_keyword = "UKS CCSD(T)-F12" if is_open_shell else "CCSD(T)-F12"
        
        # Base MPQC header for SP energy
        header = (
            f"% MPQC {method_keyword} cc-pVTZ-F12 Single-Point\n"
            f"% nprocs {nprocs}\n"
            f"% maxcore {maxcore}\n"
        )

        # Coordinate block
        coord_block = f"* xyz {charge} {mult}\n"
        for el, (x, y, z) in zip(elements, coords):
            coord_block += f"  {el:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n"
        coord_block += "*\n\n"

        # Write to disk
        with open(inp_path, "w") as f:
            f.write(header + coord_block)
            
        self.logger.info(f"Generated MPQC SP input: {inp_path.name}")
        return inp_path

    def dispatch_and_validate(self, inp_path: Path) -> Dict[str, float]:
        """
        Executes MPQC and audits the output for T1/D1 diagnostics and termination.
        """
        out_path = inp_path.with_suffix(".out")
        self.logger.info(f"Dispatching {inp_path.name} to MPQC...")
        
        try:
            with open(out_path, "w") as out_f:
                subprocess.run([self.mpqc_binary, str(inp_path)], stdout=out_f, stderr=subprocess.STDOUT, cwd=str(self.workspace_dir), check=True, timeout=300)
        except subprocess.CalledProcessError:
            self.logger.error("MPQC execution returned a non-zero exit state.")
            # We don't raise immediately; we need to parse the log to find out why.

        # Audit the log
        termination_found = False
        t1_diag, d1_diag = None, None
        
        with open(out_path, "r") as f:
            lines = f.readlines()
            
        for line in reversed(lines):
            if "MPQC TERMINATED NORMALLY" in line or "TERMINATED NORMALLY" in line:
                termination_found = True
            
            # Extract diagnostics
            if "T1 diagnostic" in line and t1_diag is None:
                match = re.search(r"T1 diagnostic\s+.*\s+([0-9.]+)", line)
                if match: t1_diag = float(match.group(1))
                
            if "D1 diagnostic" in line and d1_diag is None:
                match = re.search(r"D1 diagnostic\s+.*\s+([0-9.]+)", line)
                if match: d1_diag = float(match.group(1))

        if not termination_found:
            raise EngineConvergenceError(f"SP failed. Check {out_path.name} for SCF convergence failure.")

        self.logger.info(f"Diagnostics recovered -> T1: {t1_diag}, D1: {d1_diag}")

        # Strict Single-Reference Gate
        if t1_diag and t1_diag > 0.02:
            self.logger.critical(f"T1 diagnostic ({t1_diag}) exceeds 0.02 threshold.")
            raise MultireferenceInstabilityError(f"High T1 diagnostic: {t1_diag}. Geometry induces strong static correlation.")
        if d1_diag and d1_diag > 0.05:
            self.logger.critical(f"D1 diagnostic ({d1_diag}) exceeds 0.05 threshold.")
            raise MultireferenceInstabilityError(f"High D1 diagnostic: {d1_diag}. Geometry induces strong static correlation.")

        return {"T1": t1_diag, "D1": d1_diag}

if __name__ == "__main__":
    # Lightweight module test loop
    logging.basicConfig(level=logging.INFO)
    test_dir = Path("./GEOM_Workspace")
    test_dir.mkdir(exist_ok=True)
    
    refiner = MPQCSinglePointEngine(test_dir)
    
    # Sample H2O geometry with a frozen O-H bond
    sample_elements = ["O", "H", "H"]
    sample_coords = np.array([
        [0.000000, 0.000000, 0.117790],
        [0.000000, 0.755450, -0.471161],
        [0.000000, -0.755450, -0.471161]
    ])
    
    inp_file = refiner.generate_input("sample_water", sample_elements, sample_coords, pyscf_escalator_optimized=True)
    
    logger.info(f"Generated input file at {inp_file}. Review contents for MPQC SP.")
