#!/usr/bin/env python3
"""
CoChem-GEOM (v4.0) - Stage 3.3: Constrained CCSD(T) Refinement
--------------------------------------------------------------
Generates tightly constrained ORCA 6.1.1 inputs, freezing empirically 
derived parameters while relaxing undefined coordinates at the 
DLPNO-CCSD(T) level. Enforces T1/D1 multireference diagnostic checks.
"""

import os
import re
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

class MultireferenceInstabilityError(Exception):
    """Raised when T1 > 0.02 or D1 > 0.05, indicating wavefunction failure."""
    pass

class EngineConvergenceError(Exception):
    """Raised when ORCA fails to reach SCF or Geometry convergence."""
    pass

class ConstrainedORCAOptimizer:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.logger = logging.getLogger("CoChem_GEOM_CCSDT")
        self.orca_binary = os.environ.get("ORCA_CMD", "orca")

    def generate_input(self, base_name: str, elements: List[str], coords: np.ndarray, frozen_zmat: List[Dict], charge: int = 0, mult: int = 1) -> Path:
        """
        Constructs the ORCA 6.1.1 input file, injecting specific %geom Constraints.
        frozen_zmat expects dictionaries: {"type": "Bond", "atoms": [0, 1]}
        """
        inp_path = self.workspace_dir / f"{base_name}_ccsdt_refine.inp"
        
        # Base DLPNO-CCSD(T) header for optimization
        header = (
            f"! DLPNO-CCSD(T) def2-TZVPP def2-TZVPP/C def2/J TightSCF TightOpt\n"
            f"%pal nprocs {max(1, os.cpu_count() - 2)} end\n"
            f"%maxcore {int(4000)} # Set per core\n"
        )

        # Coordinate block
        coord_block = f"* xyz {charge} {mult}\n"
        for el, (x, y, z) in zip(elements, coords):
            coord_block += f"  {el:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n"
        coord_block += "*\n\n"

        # Constraints block
        constraint_lines = []
        for param in frozen_zmat:
            idx = param["atoms"]
            p_type = param["type"]
            if p_type == "Bond" and len(idx) == 2:
                constraint_lines.append(f"  {{ C {idx[0]} {idx[1]} C }}")
            elif p_type == "Angle" and len(idx) == 3:
                constraint_lines.append(f"  {{ A {idx[0]} {idx[1]} {idx[2]} C }}")
            elif p_type == "Dihedral" and len(idx) == 4:
                constraint_lines.append(f"  {{ D {idx[0]} {idx[1]} {idx[2]} {idx[3]} C }}")
                
        constraint_block = ""
        if constraint_lines:
            constraint_block = "%geom Constraints\n" + "\n".join(constraint_lines) + "\n  end\nend\n"

        # Write to disk
        with open(inp_path, "w") as f:
            f.write(header + constraint_block + coord_block)
            
        self.logger.info(f"Generated constrained ORCA input: {inp_path.name}")
        return inp_path

    def dispatch_and_validate(self, inp_path: Path) -> Dict[str, float]:
        """
        Executes ORCA and audits the output for T1/D1 diagnostics and termination.
        """
        out_path = inp_path.with_suffix(".out")
        self.logger.info(f"Dispatching {inp_path.name} to ORCA...")
        
        try:
            with open(out_path, "w") as out_f:
                subprocess.run(
                    [self.orca_binary, str(inp_path)],
                    stdout=out_f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.workspace_dir),
                    check=True
                )
        except subprocess.CalledProcessError:
            self.logger.error("ORCA execution returned a non-zero exit state.")
            # We don't raise immediately; we need to parse the log to find out why.

        # Audit the log
        termination_found = False
        t1_diag, d1_diag = None, None
        
        with open(out_path, "r") as f:
            lines = f.readlines()
            
        for line in reversed(lines):
            if "ORCA TERMINATED NORMALLY" in line:
                termination_found = True
            
            # Extract diagnostics
            if "T1 diagnostic" in line and t1_diag is None:
                match = re.search(r"T1 diagnostic\s+.*\s+([0-9.]+)", line)
                if match: t1_diag = float(match.group(1))
                
            if "D1 diagnostic" in line and d1_diag is None:
                match = re.search(r"D1 diagnostic\s+.*\s+([0-9.]+)", line)
                if match: d1_diag = float(match.group(1))

        if not termination_found:
            raise EngineConvergenceError(f"Optimization failed. Check {out_path.name} for SCF or geometry convergence failure.")

        self.logger.info(f"Diagnostics recovered -> T1: {t1_diag}, D1: {d1_diag}")

        # Strict Single-Reference Gate
        if t1_diag and t1_diag > 0.02:
            self.logger.critical(f"T1 diagnostic ({t1_diag}) exceeds 0.02 threshold.")
            raise MultireferenceInstabilityError(f"High T1 diagnostic: {t1_diag}. Constrained geometry induces strong static correlation.")
        if d1_diag and d1_diag > 0.05:
            self.logger.critical(f"D1 diagnostic ({d1_diag}) exceeds 0.05 threshold.")
            raise MultireferenceInstabilityError(f"High D1 diagnostic: {d1_diag}. Constrained geometry induces strong static correlation.")

        return {"T1": t1_diag, "D1": d1_diag}

if __name__ == "__main__":
    # Lightweight module test loop
    logging.basicConfig(level=logging.INFO)
    test_dir = Path("./GEOM_Workspace")
    test_dir.mkdir(exist_ok=True)
    
    refiner = ConstrainedORCAOptimizer(test_dir)
    
    # Mock H2O geometry with a frozen O-H bond
    mock_elements = ["O", "H", "H"]
    mock_coords = np.array([
        [0.000000, 0.000000, 0.117790],
        [0.000000, 0.755450, -0.471161],
        [0.000000, -0.755450, -0.471161]
    ])
    mock_frozen = [
        {"type": "Bond", "atoms": [0, 1]},
        {"type": "Angle", "atoms": [1, 0, 2]}
    ]
    
    inp_file = refiner.generate_input("mock_water", mock_elements, mock_coords, mock_frozen)
    
    print(f"Generated input file at {inp_file}. Review contents to verify strict ORCA 6.1.1 constraint syntax.")