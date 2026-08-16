#!/usr/bin/env python3
"""
CoChem-GEOM (v4.0) - 3-Tier Routing Protocol
--------------------------------------------
Implements the Method Matrix v4 routing logic for geometry optimization
and vibrational analysis:
1. Analytic CCSD(T) force fields -> CFOUR
2. Numerical / SPE / DFT -> ORCA
3. Explicit separation of CFOUR vs ORCA tracks based on requirements.
"""

import logging
from typing import Dict, Any

class EngineRouter:
    def __init__(self):
        self.logger = logging.getLogger("CoChem_GEOM_Router")
        
    def determine_route(self, task_type: str, method: str, num_atoms: int) -> str:
        """
        Implements the 3-Tier Routing Protocol.
        task_type: 'opt', 'freq', 'anharmonic', 'sp'
        method: e.g., 'CCSD(T)', 'DFT', 'MP2'
        num_atoms: number of atoms
        
        Returns the engine to use ('CFOUR', 'ORCA', or 'MPQC')
        """
        self.logger.info(f"Routing {task_type} request for {method} with {num_atoms} atoms.")
        
        if "CCSD(T)" in method.upper():
            if task_type in ["opt", "freq", "anharmonic"]:
                # CFOUR has analytic CCSD(T) second derivatives.
                # ORCA has analytic Hessians for SCF only.
                # Method Matrix explicitly routes CCSD(T) anharmonicity and analytic gradients to CFOUR.
                self.logger.info("Routing analytic CCSD(T) task to CFOUR (analytic derivatives).")
                return "CFOUR"
            elif task_type == "sp":
                # For single-points, MPQC can be used for CCSD(T)-F12.
                self.logger.info("Routing CCSD(T) SPE task to MPQC.")
                return "MPQC"
        else:
            # Numerical / SPE / DFT / DLPNO -> ORCA
            self.logger.info(f"Routing {method} task to ORCA.")
            return "ORCA"
        
        # Fallback
        return "ORCA"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    router = EngineRouter()
    print(router.determine_route("anharmonic", "CCSD(T)", 5))
    print(router.determine_route("freq", "DFT", 8))
    print(router.determine_route("sp", "CCSD(T)-F12", 6))
