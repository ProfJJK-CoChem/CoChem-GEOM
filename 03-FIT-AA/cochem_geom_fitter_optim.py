#!/usr/bin/env python3
"""CoChem-GEOM Stage 3.2: Honest Kraitchman r_s & Weighted Multi-Seed Optimizer.

Fixes:
- Preserves negative Kraitchman r_s^2 as explicit `imaginary_axis=True` Costain
  flags (never silently clips negative r_s^2 to 0.0).
- Passes `experimental_weights` directly into `scipy.optimize.least_squares`
  weighted residuals AND `J^T W J` covariance estimation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.optimize

from cochem_geom.kraitchman import KraitchmanAtomResult, solve_kraitchman_from_moments

logger = logging.getLogger(__name__)


class KraitchmanEngine:
    """Honest Kraitchman substitution solver preserving imaginary axes."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("CoChem_GEOM_Kraitchman")

    def fit_rs_kraitchman(
        self,
        parent_I: np.ndarray,
        iso_I: np.ndarray,
        M_parent: float,
        delta_m: float,
        return_report: bool = False,
        atom_label: str = "X",
    ) -> KraitchmanAtomResult | dict[str, Any]:
        result = solve_kraitchman_from_moments(
            parent_moments_u_a2=parent_I,
            iso_moments_u_a2=iso_I,
            parent_mass_u=M_parent,
            delta_mass_u=delta_m,
            atom_label=atom_label,
        )
        if return_report:
            return {
                "result": result,
                "coordinates": [result.axes[ax].abs_coord_angstrom for ax in ("a", "b", "c")],
                "rs_squared": [result.axes[ax].rs_squared_a2 for ax in ("a", "b", "c")],
                "imaginary_axis": {ax: result.axes[ax].imaginary_axis for ax in ("a", "b", "c")},
                "condition_number": result.condition_number,
            }
        return result


class MultiSeedOptimizer:
    """Multi-start weighted nonlinear least-squares optimizer with SVD covariance diagnostics."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("CoChem_GEOM_Optimizer")

    def _check_divergence(self, coords: np.ndarray, elements: list[str] | None = None) -> bool:
        from scipy.spatial.distance import pdist, squareform

        n_atoms = len(coords)
        if n_atoms < 2:
            return False
        distances = pdist(coords)
        dist_matrix = squareform(distances)
        radii_table = {
            "H": (0.31, 1.20), "HE": (0.28, 1.40), "LI": (1.28, 1.82), "BE": (0.96, 1.53),
            "B": (0.84, 1.92), "C": (0.76, 1.70), "N": (0.71, 1.55), "O": (0.66, 1.52),
            "F": (0.57, 1.47), "NE": (0.58, 1.54), "NA": (1.66, 2.27), "MG": (1.41, 1.73),
            "AL": (1.21, 1.84), "SI": (1.11, 2.10), "P": (1.07, 1.80), "S": (1.05, 1.80),
            "CL": (1.02, 1.75), "AR": (1.06, 1.88), "BR": (1.20, 1.85), "I": (1.39, 1.98),
        }
        span = float(np.max(distances)) if len(distances) > 0 else 5.0
        max_allowed_dist = max(15.0, 2.5 * span)
        for i in range(n_atoms):
            elem_i = elements[i].upper() if elements and i < len(elements) else "C"
            cov_i, _ = radii_table.get(elem_i, (0.77, 1.70))
            for j in range(i + 1, n_atoms):
                elem_j = elements[j].upper() if elements and j < len(elements) else "C"
                cov_j, _ = radii_table.get(elem_j, (0.77, 1.70))
                if dist_matrix[i, j] < 0.45 * (cov_i + cov_j) or dist_matrix[i, j] > max_allowed_dist:
                    return True
        return False

    def _generate_seeds(
        self,
        bounds: tuple[list[float], list[float]],
        n_seeds: int = 50,
        seed: int | None = None,
    ) -> np.ndarray:
        lower, upper = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
        from scipy.stats.qmc import LatinHypercube

        sampler = LatinHypercube(d=len(lower), seed=seed)
        return lower + sampler.random(n=n_seeds) * (upper - lower)

    def evaluate_jacobian_covariance(
        self, J: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w <= 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("All experimental weights must be finite and strictly positive.")
        hessian_approx = J.T @ np.diag(w) @ J
        cond_number = float(np.linalg.cond(hessian_approx))
        _, s, _ = np.linalg.svd(hessian_approx)
        truncated_mask = s / max(float(s[0]), 1e-12) < 1e-4
        truncated_count = int(np.sum(truncated_mask))
        used_svd = bool(cond_number > 1e5 or truncated_count > 0)
        covariance = (
            np.linalg.pinv(hessian_approx, rcond=1e-4)
            if used_svd
            else np.linalg.inv(hessian_approx)
        )
        svd_report = {
            "condition_number": cond_number,
            "singular_values": s.tolist(),
            "truncated_components_count": truncated_count,
            "unconstrained_variance_flag": used_svd,
        }
        return covariance, used_svd, svd_report

    def execute_fit(
        self,
        objective_fn: Callable[[np.ndarray], np.ndarray],
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | str,
        bounds: tuple[list[float], list[float]],
        experimental_weights: np.ndarray,
        n_seeds: int = 50,
        seed: int | None = 42,
    ) -> dict[str, Any]:
        w = np.asarray(experimental_weights, dtype=np.float64)
        if np.any(w <= 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("experimental_weights must be finite and strictly positive.")
        sqrt_w = np.sqrt(w)

        seeds = self._generate_seeds(bounds, n_seeds=n_seeds, seed=seed)
        best_cost = np.inf
        best_seed = seeds[0]
        for s_vec in seeds:
            try:
                res = np.asarray(objective_fn(s_vec), dtype=np.float64)
                cost = float(np.sum((res ** 2) * w))
                if cost < best_cost:
                    best_cost = cost
                    best_seed = s_vec
            except Exception:
                continue

        def weighted_obj(p: np.ndarray) -> np.ndarray:
            return sqrt_w * np.asarray(objective_fn(p), dtype=np.float64)

        if callable(jacobian_fn):
            def weighted_jac(p: np.ndarray) -> np.ndarray:
                return sqrt_w[:, np.newaxis] * np.asarray(jacobian_fn(p), dtype=np.float64)
            jac_arg: Any = weighted_jac
        else:
            jac_arg = jacobian_fn

        result = scipy.optimize.least_squares(
            fun=weighted_obj,
            x0=best_seed,
            jac=jac_arg,
            bounds=bounds,
            method="trf",
            x_scale="jac",
            loss="linear",
            tr_solver="exact",
        )
        if not result.success:
            raise RuntimeError(f"Geometry fitter diverged: {result.message}")

        j_unweighted = np.asarray(result.jac, dtype=np.float64) / sqrt_w[:, np.newaxis]
        covariance, svd_triggered, svd_report = self.evaluate_jacobian_covariance(j_unweighted, w)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        return {
            "converged_parameters": result.x,
            "standard_errors": standard_errors,
            "covariance_matrix": covariance,
            "svd_fallback_used": svd_triggered,
            "svd_report": svd_report,
            "cost": float(result.cost),
            "optimality": float(result.optimality),
        }
