# **Stage 3.1 — Internal Coordinate & Quaternion Engine**

**Target File:** cochem\_geom\_fitter\_core.py

### **1\. Purpose & Scope**

Maps the Cartesian space to stable Z-matrix forms for optimization, computes the internal B-matrix, and dynamically sets bounds. **Must be completely independent of GUI logic.**

### **2\. Required Imports & Dependencies**

`import numpy as np`  
`from scipy.spatial.transform import Rotation`  
`from mendeleev import element`

### **3\. Core Logic**

**Class: ZMatrixEngine**

> * **Method: \_compute\_b\_matrix(self, cartesian\_coords: np.ndarray, z\_matrix\_def: list) \-\> np.ndarray**  
  * Computes the Wilson B-matrix (B\_{ij} \= \\frac{\\partial q\_i}{\\partial x\_j}).  
  * Essential for projecting Cartesian gradients into the internal coordinate space. Uses finite difference (10^{-5} Å) if analytical derivatives become singular (e.g., linear angles).  
> * **Method: \_apply\_eckart\_quaternion(self, ref\_coords: np.ndarray, new\_coords: np.ndarray) \-\> np.ndarray**  
  * Prevents unphysical rigid-body rotation during Z-matrix flexing.  
  * Solves the Kabsch SVD problem between new\_coords and ref\_coords, extracts the rotation matrix, converts it to a Quaternion via scipy.spatial.transform.Rotation, and counter-rotates new\_coords.

**Class: DynamicBoundsTuner**

> * **Method: get\_bond\_bounds(self, atom\_A: str, atom\_B: str) \-\> tuple**  
  * Looks up element(atom\_A).covalent\_radius and element(atom\_B).covalent\_radius.  
  * Base bond length \= sum of radii.  
  * Returns (base \* 0.8, base \* 1.25). The optimizer is strictly forbidden from breaking these bounds.