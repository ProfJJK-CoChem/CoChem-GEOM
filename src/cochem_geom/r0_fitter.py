"""Weighted multi-isotopologue r_0 (and optional r_e^SE) nonlinear least-squares fitter.

Key scientific safeguards:
- Strictly rejects missing, None, non-finite, or <= 0 uncertainties with `ValueError`.
- Passes experimental weights w_i = 1 / sigma_i^2 directly into
  `scipy.optimize.least_squares` residuals:
      r_i(theta) = sqrt(w_i) * (B_i^calc(theta) - B_i^obs)
  so that changing weights deterministically changes the converged parameters.
- Evaluates the exact weighted covariance matrix Sigma = (J^T W J)^-1 (as well as
  reduced-chi^2 scaled covariance s^2 (J^T W J)^-1) using the unweighted model
  Jacobian J_unw and W = diag(w_i), with SVD condition-number diagnostics.
- Uses single CODATA conversion factor `ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2 = 505379.008435`.
- Explicitly separates r_0 (fitted to experimental B_0) from r_e^SE (only populated
  when verified isotope-specific vibrational corrections Delta_B_vib are supplied;
  otherwise explicitly reported as PENDING).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import scipy.optimize
from pydantic import BaseModel, ConfigDict, field_validator

from .constants import get_exact_isotope_mass
from .inertia import rotational_constants_from_geometry


class IsotopologueObservation(BaseModel):
    """Experimental rotational constants (A0, B0, C0) and strict positive uncertainties."""

    model_config = ConfigDict(extra="forbid")

    species_id: str
    symbols: list[str]
    masses_u: list[float] | None = None
    rotational_constants_mhz: tuple[float, float, float]
    uncertainties_mhz: tuple[float, float, float]
    vibrational_corrections_mhz: tuple[float, float, float] | None = None
    source_citation: str = ""

    @field_validator("rotational_constants_mhz")
    @classmethod
    def _check_positive_constants(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        for idx, val in enumerate(v):
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"Rotational constant index {idx} must be finite and > 0, got {val!r}")
        return v

    @field_validator("uncertainties_mhz")
    @classmethod
    def _check_positive_uncertainties(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        for idx, val in enumerate(v):
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(
                    f"Uncertainty index {idx} must be finite and strictly positive, got {val!r}; "
                    "silent default to 0.0 is prohibited."
                )
        return v

    def resolved_masses_u(self) -> np.ndarray:
        if self.masses_u is not None:
            if len(self.masses_u) != len(self.symbols):
                raise ValueError("masses_u length must match symbols length.")
            arr = np.asarray(self.masses_u, dtype=np.float64)
            if np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
                raise ValueError("All masses_u must be finite and positive.")
            return arr
        return np.array([get_exact_isotope_mass(s) for s in self.symbols], dtype=np.float64)


class ResidualRecord(BaseModel):
    """Per-observable fit residual."""

    model_config = ConfigDict(extra="forbid")

    species_id: str
    axis: Literal["A", "B", "C"]
    observed_mhz: float
    calculated_mhz: float
    uncertainty_mhz: float
    weight_mhz_inv2: float
    residual_mhz: float
    weighted_residual: float


class R0FitResult(BaseModel):
    """Verified weighted nonlinear least-squares r_0 (or r_e^SE) fit output."""

    model_config = ConfigDict(extra="forbid")

    structure_type: Literal["r0", "re_se"]
    parameter_names: list[str]
    parameter_units: list[str]
    fitted_values: dict[str, float]
    unscaled_standard_errors: dict[str, float]
    scaled_standard_errors: dict[str, float]
    covariance_unscaled: list[list[float]]
    covariance_scaled: list[list[float]]
    correlation_matrix: list[list[float]]
    n_observations: int
    n_parameters: int
    degrees_of_freedom: int
    chi_squared: float
    reduced_chi_squared: float
    rms_residual_mhz: float
    weighted_rms: float
    condition_number: float
    svd_fallback_used: bool
    converged: bool
    optimizer_message: str
    residuals: list[ResidualRecord]


def build_c2v_triatomic_geometry(params: np.ndarray) -> np.ndarray:
    """Construct C2v triatomic XY2 Cartesian coordinates (Angstrom) from [r_XY (A), angle_YXY (deg)].

    Atom order: [central X, terminal Y1, terminal Y2].
    """
    r_xy = float(params[0])
    angle_deg = float(params[1])
    half_rad = math.radians(angle_deg) / 2.0
    dx = r_xy * math.sin(half_rad)
    dy = r_xy * math.cos(half_rad)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [dx, dy, 0.0],
            [-dx, dy, 0.0],
        ],
        dtype=np.float64,
    )


def fit_weighted_structure(
    isotopologues: list[IsotopologueObservation],
    geometry_builder: Callable[[np.ndarray], np.ndarray],
    initial_params: list[float] | np.ndarray,
    parameter_names: list[str],
    parameter_units: list[str],
    bounds: tuple[list[float], list[float]],
    *,
    structure_type: Literal["r0", "re_se"] = "r0",
    ftol: float = 1e-12,
    xtol: float = 1e-12,
    gtol: float = 1e-12,
) -> R0FitResult:
    """Perform weighted multi-isotopologue nonlinear least-squares structural fit.

    Weights `w_i = 1 / sigma_i^2` are passed directly into `scipy.optimize.least_squares`
    as `sqrt(w_i) * (B_calc - B_obs)` AND used in `J^T W J` covariance estimation.
    """
    if not isotopologues:
        raise ValueError("At least one isotopologue observation is required for structural fitting.")

    x0 = np.asarray(initial_params, dtype=np.float64)
    n_params = int(x0.size)
    if len(parameter_names) != n_params or len(parameter_units) != n_params:
        raise ValueError("parameter_names and parameter_units must match initial_params length.")

    obs_values: list[float] = []
    obs_sigmas: list[float] = []
    obs_meta: list[tuple[str, Literal["A", "B", "C"]]] = []
    iso_masses: list[np.ndarray] = []

    axes: tuple[Literal["A", "B", "C"], ...] = ("A", "B", "C")
    for iso in isotopologues:
        masses = iso.resolved_masses_u()
        iso_masses.append(masses)
        base_b = np.asarray(iso.rotational_constants_mhz, dtype=np.float64)
        if structure_type == "re_se":
            if iso.vibrational_corrections_mhz is None:
                raise ValueError(
                    f"r_e^SE fit requested, but vibrational_corrections_mhz is missing for {iso.species_id}. "
                    "Leave r_e^SE PENDING unless verified VPT2 Delta_B_vib values are supplied."
                )
            # B_e^SE = B_0^exp - Delta_B_vib (where Delta_B_vib = -1/2 sum alpha_r)
            target_b = base_b - np.asarray(iso.vibrational_corrections_mhz, dtype=np.float64)
        else:
            target_b = base_b

        for ax_idx, ax_name in enumerate(axes):
            obs_values.append(float(target_b[ax_idx]))
            obs_sigmas.append(float(iso.uncertainties_mhz[ax_idx]))
            obs_meta.append((iso.species_id, ax_name))

    obs_arr = np.asarray(obs_values, dtype=np.float64)
    sig_arr = np.asarray(obs_sigmas, dtype=np.float64)
    if np.any(sig_arr <= 0.0) or not np.all(np.isfinite(sig_arr)):
        raise ValueError("All experimental uncertainties must be finite and strictly positive.")

    weights = 1.0 / (sig_arr ** 2)
    sqrt_weights = 1.0 / sig_arr
    n_obs = int(obs_arr.size)
    if n_obs < n_params:
        raise ValueError(
            f"Underdetermined fit: {n_obs} rotational observations for {n_params} geometric parameters."
        )

    def predict_all(theta: np.ndarray) -> np.ndarray:
        coords = geometry_builder(theta)
        preds: list[float] = []
        for masses in iso_masses:
            _, rot_mhz = rotational_constants_from_geometry(coords, masses)
            preds.extend(rot_mhz.tolist())
        return np.asarray(preds, dtype=np.float64)

    def weighted_residuals(theta: np.ndarray) -> np.ndarray:
        return sqrt_weights * (predict_all(theta) - obs_arr)

    res = scipy.optimize.least_squares(
        fun=weighted_residuals,
        x0=x0,
        bounds=bounds,
        method="trf",
        x_scale="jac",
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=2000,
    )
    if not res.success:
        raise RuntimeError(f"Weighted {structure_type} optimization failed to converge: {res.message}")

    theta_hat = res.x
    calc_arr = predict_all(theta_hat)
    raw_resid = calc_arr - obs_arr
    w_resid = sqrt_weights * raw_resid

    # res.jac is J_w = diag(sqrt(w)) @ J_unweighted.
    # Recover unweighted J so H = J_unw^T @ W @ J_unw == J_w^T @ J_w identically.
    j_weighted = np.asarray(res.jac, dtype=np.float64)
    j_unweighted = j_weighted / sqrt_weights[:, np.newaxis]
    w_mat = np.diag(weights)
    hessian_approx = j_unweighted.T @ w_mat @ j_unweighted

    cond_num = float(np.linalg.cond(hessian_approx))
    _, svals, _ = np.linalg.svd(hessian_approx)
    svd_fallback = bool(cond_num > 1e10 or np.any(svals / max(float(svals[0]), 1e-15) < 1e-8))
    if svd_fallback:
        cov_unscaled = np.linalg.pinv(hessian_approx, rcond=1e-8)
    else:
        cov_unscaled = np.linalg.inv(hessian_approx)

    dof = max(n_obs - n_params, 1)
    chi2 = float(np.sum(w_resid ** 2))
    red_chi2 = chi2 / float(dof)
    cov_scaled = cov_unscaled * red_chi2

    se_unscaled = np.sqrt(np.maximum(np.diag(cov_unscaled), 0.0))
    se_scaled = np.sqrt(np.maximum(np.diag(cov_scaled), 0.0))

    denom_corr = np.outer(se_unscaled, se_unscaled)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom_corr > 0.0, cov_unscaled / denom_corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)

    residual_records = [
        ResidualRecord(
            species_id=obs_meta[i][0],
            axis=obs_meta[i][1],
            observed_mhz=float(obs_arr[i]),
            calculated_mhz=float(calc_arr[i]),
            uncertainty_mhz=float(sig_arr[i]),
            weight_mhz_inv2=float(weights[i]),
            residual_mhz=float(raw_resid[i]),
            weighted_residual=float(w_resid[i]),
        )
        for i in range(n_obs)
    ]

    return R0FitResult(
        structure_type=structure_type,
        parameter_names=list(parameter_names),
        parameter_units=list(parameter_units),
        fitted_values={name: float(theta_hat[i]) for i, name in enumerate(parameter_names)},
        unscaled_standard_errors={name: float(se_unscaled[i]) for i, name in enumerate(parameter_names)},
        scaled_standard_errors={name: float(se_scaled[i]) for i, name in enumerate(parameter_names)},
        covariance_unscaled=cov_unscaled.tolist(),
        covariance_scaled=cov_scaled.tolist(),
        correlation_matrix=corr.tolist(),
        n_observations=n_obs,
        n_parameters=n_params,
        degrees_of_freedom=n_obs - n_params,
        chi_squared=chi2,
        reduced_chi_squared=red_chi2,
        rms_residual_mhz=float(np.sqrt(np.mean(raw_resid ** 2))),
        weighted_rms=float(np.sqrt(np.mean(w_resid ** 2))),
        condition_number=cond_num,
        svd_fallback_used=svd_fallback,
        converged=bool(res.success),
        optimizer_message=str(res.message),
        residuals=residual_records,
    )


def fit_c2v_triatomic_r0(
    isotopologues: list[IsotopologueObservation],
    *,
    initial_r_ang: float = 0.96,
    initial_angle_deg: float = 104.5,
    r_bounds: tuple[float, float] = (0.5, 2.5),
    angle_bounds: tuple[float, float] = (60.0, 179.0),
    structure_type: Literal["r0", "re_se"] = "r0",
    bond_label: str = "r(O-H)",
    angle_label: str = "angle(H-O-H)",
) -> R0FitResult:
    """Convenience wrapper fitting weighted r_0 (or r_e^SE) for a C2v triatomic XY2."""
    return fit_weighted_structure(
        isotopologues=isotopologues,
        geometry_builder=build_c2v_triatomic_geometry,
        initial_params=[initial_r_ang, initial_angle_deg],
        parameter_names=[bond_label, angle_label],
        parameter_units=["Angstrom", "deg"],
        bounds=([r_bounds[0], angle_bounds[0]], [r_bounds[1], angle_bounds[1]]),
        structure_type=structure_type,
    )


def format_r0_summary(result: R0FitResult) -> dict[str, Any]:
    """Return concise dictionary summary of fitted parameters and uncertainties."""
    return {
        "structure_type": result.structure_type,
        "parameters": {
            name: {
                "value": result.fitted_values[name],
                "unit": result.parameter_units[idx],
                "std_err_scaled": result.scaled_standard_errors[name],
                "std_err_unscaled": result.unscaled_standard_errors[name],
            }
            for idx, name in enumerate(result.parameter_names)
        },
        "rms_residual_mhz": result.rms_residual_mhz,
        "chi_squared": result.chi_squared,
        "reduced_chi_squared": result.reduced_chi_squared,
        "degrees_of_freedom": result.degrees_of_freedom,
        "condition_number": result.condition_number,
    }
