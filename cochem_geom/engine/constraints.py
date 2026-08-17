import numpy as np
from typing import List, Tuple

class ConstraintEngine:
    def __init__(self):
        self.frozen_bonds: List[Tuple[int, int, float]] = []

    def freeze_bond(self, i: int, j: int, distance: float):
        self.frozen_bonds.append((i, j, float(distance)))

    def apply_gradient_projection(self, geometry: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        corr_grad = gradients.copy()
        for i, j, _ in self.frozen_bonds:
            diff = geometry[j] - geometry[i]
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                continue
            u = diff / norm
            # Remove component of g_i and g_j along u so that np.dot(g_i, u) == 0 and np.dot(g_j, u) == 0
            g_i_proj = np.dot(corr_grad[i], u)
            g_j_proj = np.dot(corr_grad[j], u)
            corr_grad[i] -= g_i_proj * u
            corr_grad[j] -= g_j_proj * u
        return corr_grad

    def enforce_constraints(self, geometry: np.ndarray) -> np.ndarray:
        new_geom = geometry.copy()
        for i, j, target_dist in self.frozen_bonds:
            diff = new_geom[j] - new_geom[i]
            current_dist = np.linalg.norm(diff)
            if current_dist < 1e-12:
                continue
            u = diff / current_dist
            midpoint = 0.5 * (new_geom[i] + new_geom[j])
            new_geom[i] = midpoint - 0.5 * target_dist * u
            new_geom[j] = midpoint + 0.5 * target_dist * u
        return new_geom
