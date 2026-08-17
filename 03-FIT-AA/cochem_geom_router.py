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
from typing import Literal

class EngineRoutingError(Exception):
    pass

class UnsupportedMethodError(EngineRoutingError):
    pass

EngineType = Literal["CFOUR", "ORCA", "MPQC"]

class EngineRouter:
    def __init__(self) -> None:
        self.logger = logging.getLogger("CoChem_GEOM_Router")
        
    def determine_route(self, task_type: str, method: str, num_atoms: int) -> EngineType:
        """
        Implements the 3-Tier Routing Protocol.
        task_type: 'opt', 'freq', 'anharmonic', 'sp'
        method: e.g., 'CCSD(T)', 'DFT', 'MP2'
        num_atoms: number of atoms
        
        Returns the engine to use ('CFOUR', 'ORCA', or 'MPQC')
        """
        if not task_type or not method:
            raise UnsupportedMethodError("Task type and method must be provided.")
            
        task_type = task_type.lower()
        method = method.upper()
        
        if task_type not in ["opt", "freq", "anharmonic", "sp"]:
            raise EngineRoutingError(f"Unsupported task_type: {task_type}")
            
        if num_atoms <= 0:
            raise EngineRoutingError(f"Invalid num_atoms: {num_atoms}")
            
        self.logger.info(f"Routing {task_type} request for {method} with {num_atoms} atoms.")
        
        if "CCSD(T)" in method:
            if "DLPNO" in method:
                self.logger.info("Routing DLPNO-CCSD(T) task to ORCA.")
                return "ORCA"
                
            if task_type in ["opt", "freq", "anharmonic"]:
                if num_atoms > 16:
                    self.logger.info("Atom count > 16. Fallback to ORCA for numerical derivatives.")
                    return "ORCA"
                self.logger.info("Routing analytic CCSD(T) task to CFOUR (analytic derivatives).")
                return "CFOUR"
            elif task_type == "sp":
                self.logger.info("Routing CCSD(T) SPE task to MPQC.")
                return "MPQC"
        else:
            self.logger.info(f"Routing {method} task to ORCA.")
            return "ORCA"
            
        return "ORCA"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    router = EngineRouter()
    print(router.determine_route("anharmonic", "CCSD(T)", 5))
    print(router.determine_route("freq", "DFT", 8))
    print(router.determine_route("sp", "CCSD(T)-F12", 6))
