"""CoChem-BASE `cochem.modules` entry point provider (`geom_provider`).

Side-effect-free discovery and versioned operation registry for Phase 2
`CoChem-BASE` integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ccsd_driver import (
    parse_orca_ccsd_output,
    parse_pyscf_ccsd_output,
    run_orca_ccsd_optimization,
    run_pyscf_ccsd_optimization,
)
from .constants import ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2
from .kraitchman import solve_kraitchman_from_rotational_constants
from .pickett import (
    parse_pickett_fit,
    parse_pickett_lin,
    parse_pickett_par,
    parse_pickett_var,
    parse_spycfit_json,
)
from .r0_fitter import IsotopologueObservation, fit_c2v_triatomic_r0

INTEGRATION_CONTRACT = "cochem.module-handoff/1"
OPERATIONS: tuple[str, ...] = (
    "parse_pickett",
    "fit_kraitchman_rs",
    "fit_weighted_r0",
    "optimize_ccsd",
    "compare_structures",
)


def metadata() -> dict[str, Any]:
    """Return side-effect-free CoChem-BASE module capability metadata."""
    return {
        "module_id": "geom",
        "distribution": "CoChem-GEOM",
        "version": "0.2.0",
        "integration_contract": INTEGRATION_CONTRACT,
        "operations": list(OPERATIONS),
        "codata_rotational_factor_mhz_u_a2": ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2,
        "supported_ccsd_backends": ["pyscf_ccsd", "orca_ccsd"],
        "supported_pickett_formats": [".var", ".par", ".fit", ".lin", ".json"],
    }


def dispatch_operation(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a declared canonical CoChem-GEOM operation."""
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported CoChem-GEOM operation {operation!r}; valid: {OPERATIONS}")

    if operation == "parse_pickett":
        path = Path(payload["path"])
        suffix = path.suffix.lower()
        if suffix == ".var":
            return parse_pickett_var(path).model_dump(mode="json")
        if suffix == ".par":
            return parse_pickett_par(path).model_dump(mode="json")
        if suffix == ".fit":
            return parse_pickett_fit(path).model_dump(mode="json")
        if suffix == ".lin":
            return {"lines": [ln.model_dump(mode="json") for ln in parse_pickett_lin(path)]}
        if suffix == ".json":
            return parse_spycfit_json(path).model_dump(mode="json")
        raise ValueError(f"Unsupported Pickett file suffix: {suffix!r}")

    if operation == "fit_kraitchman_rs":
        res = solve_kraitchman_from_rotational_constants(
            parent_abc_mhz=payload["parent_abc_mhz"],
            iso_abc_mhz=payload["iso_abc_mhz"],
            parent_mass_u=float(payload["parent_mass_u"]),
            delta_mass_u=float(payload["delta_mass_u"]),
            atom_label=str(payload.get("atom_label", "X")),
            parent_unc_mhz=payload.get("parent_unc_mhz"),
            iso_unc_mhz=payload.get("iso_unc_mhz"),
        )
        return res.model_dump(mode="json")

    if operation == "fit_weighted_r0":
        isos = [IsotopologueObservation.model_validate(item) for item in payload["isotopologues"]]
        res = fit_c2v_triatomic_r0(
            isos,
            initial_r_ang=float(payload.get("initial_r_ang", 0.96)),
            initial_angle_deg=float(payload.get("initial_angle_deg", 104.5)),
            structure_type=payload.get("structure_type", "r0"),
        )
        return res.model_dump(mode="json")

    if operation == "optimize_ccsd":
        backend = payload.get("backend", "pyscf_ccsd")
        if payload.get("parse_only_output"):
            out_path = Path(payload["parse_only_output"])
            if backend == "orca_ccsd":
                return parse_orca_ccsd_output(out_path).model_dump(mode="json")
            if backend == "pyscf_ccsd":
                return parse_pyscf_ccsd_output(out_path).model_dump(mode="json")
            raise ValueError(f"Unknown CCSD backend: {backend!r}")
        if backend == "orca_ccsd":
            return run_orca_ccsd_optimization(
                work_dir=payload["work_dir"],
                job_name=payload["job_name"],
                symbols=payload["symbols"],
                initial_coords_ang=payload["initial_coords_ang"],
                basis=payload.get("basis", "cc-pVTZ"),
                charge=int(payload.get("charge", 0)),
                multiplicity=int(payload.get("multiplicity", 1)),
            ).model_dump(mode="json")
        if backend == "pyscf_ccsd":
            return run_pyscf_ccsd_optimization(
                work_dir=payload["work_dir"],
                job_name=payload["job_name"],
                symbols=payload["symbols"],
                initial_coords_ang=payload["initial_coords_ang"],
                basis=payload.get("basis", "cc-pVDZ"),
                charge=int(payload.get("charge", 0)),
                multiplicity=int(payload.get("multiplicity", 1)),
            ).model_dump(mode="json")
        raise ValueError(f"Unknown CCSD backend: {backend!r}")

    raise ValueError("compare_structures must be invoked via CLI or reporting.build_structural_comparison_payload")


class GeomProvider:
    """CoChem-BASE `cochem.modules` provider object for CoChem-GEOM."""

    module_id = "geom"
    distribution = "CoChem-GEOM"
    metadata = staticmethod(metadata)
    dispatch_operation = staticmethod(dispatch_operation)

    def __call__(self) -> dict[str, Any]:
        return metadata()


geom_provider = GeomProvider()
