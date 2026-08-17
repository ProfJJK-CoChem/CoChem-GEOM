import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class WebGLVectorViewer:
    def __init__(self):
        pass

    def dispatch_browser(self, geometry, hessian, masses, output_image: str) -> bool:
        geom = np.array(geometry, dtype=float)
        H = np.array(hessian, dtype=float)
        M = np.array(masses, dtype=float)

        # Mass-weight the Hessian
        N = len(geom)
        M_vec = np.repeat(np.sqrt(M), 3)
        M_inv = np.diag(1.0 / M_vec)
        H_mw = M_inv @ H @ M_inv

        eigenvalues, eigenvectors = np.linalg.eigh(H_mw)
        # Sort
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Mode 1 vector
        mode_vec = eigenvectors[:, 0].reshape((N, 3))

        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')

        # Plot atoms
        ax.scatter(geom[:, 0], geom[:, 1], geom[:, 2], color='#38bdf8', s=200, edgecolors='white', linewidths=1.5, label='Atoms')

        # Plot normal mode displacement vectors
        ax.quiver(
            geom[:, 0], geom[:, 1], geom[:, 2],
            mode_vec[:, 0], mode_vec[:, 1], mode_vec[:, 2],
            color='#f43f5e', length=0.8, normalize=True, linewidth=2.5, label='Mode Vectors'
        )

        ax.set_title("Vibrational Normal Mode Vectors (WebGL Field Dispatch)", color='white', fontsize=14, pad=20)
        ax.set_xlabel("X (Å)", color='white')
        ax.set_ylabel("Y (Å)", color='white')
        ax.set_zlabel("Z (Å)", color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
        plt.tight_layout()

        plt.savefig(output_image, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

        return os.path.exists(output_image) and os.path.getsize(output_image) > 0
