# **Stage 4.0 — Correlation Filters & Publication Generator**

**Target File:** cochem\_geom\_reporter\_latex.py

### **1\. Purpose & Scope**

Transmute converged physics data into high-fidelity academic reports using LaTeX and generate legacy Pickett files.

### **2\. Required Imports & Dependencies**

`import json`  
`import numpy as np`  
`from pathlib import Path`

### **3\. Reporting Utilities**

**Method: export\_siunitx\_table(self, fit\_results: dict) \-\> str**

> * **Logic:** Iterates over the fit parameters, sorting by Atomic Weight (Bonds \> Angles \> Dihedrals).  
> * **Format:** Constructs a valid LaTeX string using \\begin{table}, \\usepackage{siunitx}, and \\begin{tabular}{l S\[table-format=1.4(2)\] ...}.  
> * **Uncertainty Guard:** If a parameter was flagged as is\_frozen=True, the printed uncertainty value is overridden with \\text{Set Value} to denote an assumed parameter mathematically.

**Method: generate\_methods\_boilerplate(self) \-\> str**

> * **Logic:** Uses simple python f-strings to stitch together a methods section based on the executed fit\_type (e.g., r\_0 vs r\_e^{SE}).

**Method: export\_pickett\_par(self, zmat: list, rot\_consts: list) \-\> Path**

> * **Logic:** Writes a strict Fortran-compatible ASCII text file (.par) mapped for SPFIT ingestion, assigning the proper A, B, and C rigid rotor identifiers.