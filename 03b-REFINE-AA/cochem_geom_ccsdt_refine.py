#!/usr/bin/env python3
"""CoChem-GEOM Stage 3.3: Canonical CCSD Geometry Optimization Driver (ORCA 6.1.1 & PySCF CPU).

Delegates to `cochem_geom.ccsd_driver`. Strictly forbids DLPNO-CCSD(T) geometry
optimization (ORCA MDCI lacks analytic gradients) and single-point faking.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from cochem_geom.ccsd_driver import (
    CCSDConvergenceError,
    CCSDOptimizationResult,
    generate_orca_ccsd_input,
    parse_orca_ccsd_output,
    parse_pyscf_ccsd_output,
    run_orca_ccsd_optimization,
    run_pyscf_ccsd_optimization,
)


class CoChemSystemConfig(BaseModel):
    nprocs: int = Field(default_factory=lambda: int(os.environ.get("COCHEM_NPROCS", 1)))
    maxcore: int = Field(default_factory=lambda: int(os.environ.get("COCHEM_MAXCORE", 2000)))
    artifacts_dir: str = Field(
        default_factory=lambda: os.environ.get("COCHEM_ARTIFACT_DIR", str(Path.home() / "cochem_artifacts"))
    )


class MultireferenceInstabilityError(Exception):
    pass


class EngineConvergenceError(CCSDConvergenceError):
    pass


class SpinContaminationError(Exception):
    pass


class DispersionMissingError(Exception):
    pass


class CanonicalCCSDORCAOptimizer:
    """Canonical ORCA 6.1.1 CCSD TightOpt Freq geometry optimizer."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("CoChem_GEOM_ORCA_CCSD")
        self.config = CoChemSystemConfig(artifacts_dir=str(self.workspace_dir))

    def generate_input(
        self,
        base_name: str,
        elements: list[str],
        coords: np.ndarray,
        charge: int = 0,
        mult: int = 1,
        basis: str = "cc-pVTZ",
        method: str = "CCSD",
    ) -> Path:
        inp_path = self.workspace_dir / f"{base_name}.inp"
        return generate_orca_ccsd_input(
            inp_path=inp_path,
            symbols=elements,
            coords_ang=coords,
            basis=basis,
            charge=charge,
            multiplicity=mult,
            nprocs=self.config.nprocs,
            maxcore_mb=self.config.maxcore,
            method=method,
        )

    def run_and_validate(
        self,
        base_name: str,
        elements: list[str],
        coords: np.ndarray,
        charge: int = 0,
        mult: int = 1,
        basis: str = "cc-pVTZ",
    ) -> CCSDOptimizationResult:
        return run_orca_ccsd_optimization(
            work_dir=self.workspace_dir,
            job_name=base_name,
            symbols=elements,
            initial_coords_ang=coords,
            basis=basis,
            charge=charge,
            multiplicity=mult,
            nprocs=self.config.nprocs,
            maxcore_mb=self.config.maxcore,
        )

    def parse_output(self, out_path: Path) -> CCSDOptimizationResult:
        return parse_orca_ccsd_output(out_path)


class PySCFCCSDOptimizer:
    """CPU PySCF CCSD analytic-gradient + geomeTRIC optimizer."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("CoChem_GEOM_PySCF_CCSD")

    def run_and_validate(
        self,
        base_name: str,
        elements: list[str],
        coords: np.ndarray,
        charge: int = 0,
        mult: int = 1,
        basis: str = "cc-pVDZ",
    ) -> CCSDOptimizationResult:
        return run_pyscf_ccsd_optimization(
            work_dir=self.workspace_dir,
            job_name=base_name,
            symbols=elements,
            initial_coords_ang=coords,
            basis=basis,
            charge=charge,
            multiplicity=mult,
        )

    def parse_output(self, log_path: Path) -> CCSDOptimizationResult:
        return parse_pyscf_ccsd_output(log_path)


# Backwards-compatible alias
ConstrainedORCAOptimizer = CanonicalCCSDORCAOptimizer

__all__ = [
    "CoChemSystemConfig",
    "CanonicalCCSDORCAOptimizer",
    "ConstrainedORCAOptimizer",
    "PySCFCCSDOptimizer",
    "EngineConvergenceError",
    "MultireferenceInstabilityError",
    "SpinContaminationError",
    "DispersionMissingError",
]
