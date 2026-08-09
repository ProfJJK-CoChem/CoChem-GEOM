# **Stage 1.1 — Isotope Math, Axis Detection & Fingerprinting**

**Target File:** cochem\_geom\_ingest\_math.py

### **1\. Purpose & Scope**

Standardize the mass matrices using exact IUPAC values via mendeleev and physically align all ingested atomic coordinates.

### **2\. Required Imports & Dependencies**

`import numpy as np`  
`from mendeleev import element, isotope`  
`import hashlib`  
`import copy`

### **3\. Core Class: CoordinateStandardizer**

**Method: \_fetch\_exact\_mass(self, symbol: str, mass\_num: int \= None) \-\> float**

> * **Logic:** If mass\_num is None, return standard atomic weight. If defined, use mendeleev.isotope(symbol, mass\_num).mass to get the *exact* monoisotopic mass in Daltons.  
> * **Validation:** Explicitly verify ^{13}\\text{C} evaluates to exactly 13.003354835.

**Method: translate\_to\_com(self, coords: np.ndarray, masses: np.ndarray) \-\> np.ndarray**

> * **Logic:** Standard Center of Mass transformation: r\_i' \= r\_i \- \\frac{\\sum m\_i r\_i}{\\sum m\_i}.

**Method: align\_to\_principal\_axes(self, coords: np.ndarray, masses: np.ndarray) \-\> tuple\[np.ndarray, np.ndarray\]**

> * **Logic:** 1\. Build the 3 \\times 3 inertial tensor I. 2\. Diagonalize using np.linalg.eigh(I). 3\. Sort eigenvalues so that I\_a \\le I\_b \\le I\_c. 4\. Rotate coordinates into the eigenvector frame.  
> * **Dipole Flip Trap:** Store the rotation matrix R. If tracking multiple isotopologues, verify the determinant \\det(R\_0^T R\_{\\text{iso}}). If axis reorientation (a \\leftrightarrow b) occurs due to isotopic substitution shifting the principal frame, apply a permutation matrix to re-index.

**Method: fingerprint\_payload(self, raw\_data\_dict: dict) \-\> str**

> * **Logic:** Serialize the coordinates and exact masses into a UTF-8 string and compute the hashlib.sha256().hexdigest(). Append to geom\_provenance.json.