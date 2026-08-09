# **Stage 3.2 — GPU Multi-Seed Optimizer & MCMC Error Prop**

**Target File:** cochem\_geom\_fitter\_optim.py

### **1\. Purpose & Scope**

The core physics engine. Executes multi-seed least-squares fitting, leverages sparse Jacobians, dynamically triggers SVD fallback, and propagates analytical error.

### **2\. Required Imports & Dependencies**

import numpy as np  
import scipy.optimize  
import scipy.sparse as sparse  
try:  
    import torch  
except ImportError:  
    torch \= None \# Fallback to numpy

### **3\. Structural Fitting Routines**

**Method: fit\_rs\_kraitchman(self, parent\_I: np.ndarray, iso\_I: np.ndarray, delta\_m: float) \-\> tuple**

> * **Logic:** Implements standard Kraitchman equations. \\vert{}x\\vert{} \= \\sqrt{\\frac{\\Delta P\_x}{\\mu}} where \\mu \= \\frac{M \\Delta m}{M \+ \\Delta m}.  
> * **Singularity Trap:** If \\vert{}x\\vert{} \< 0.15 Å, the coordinate is dangerously close to the axis. Triggers FallbackToDoubleSubstitution or issues a warning for imaginary coordinates.

**Class: MultiSeedOptimizer**

> * **Method: execute\_fit(self, initial\_guess: np.ndarray, ...)**  
  * Generates N=50 random initial parameter seeds distributed within the Dynamic Bounds defined in 3.1.  
  * Uses scipy.optimize.least\_squares with method='trf' (Trust Region Reflective).  
  * **Condition Check:** Evaluates Jacobian J. If np.linalg.cond(J) \> 1e5, the matrix is singular. Abort trf and fall back to SVD (np.linalg.pinv(J, rcond=1e-4)).  
  * **Analytical Error:** The variance-covariance matrix is computed as \\Sigma \= (J^T W J)^{-1}, where W is the diagonal weight matrix of experimental uncertainties.  
  * Commits final converged arrays to the SWMR HDF5 lake.