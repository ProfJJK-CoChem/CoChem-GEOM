# **Stage 2.1 — Symmetry Detection & Thermal Warnings**

**Target File:** cochem\_geom\_eval\_sym.py

### **1\. Purpose & Scope**

Define point-group symmetries for all configurations, present the HITL symmetry override UI, and mathematically check for apparent non-planarity caused by zero-point vibrational motions.

### **2\. Required Imports & Dependencies**

`import molsym`  
`import numpy as np`  
`import ipywidgets as widgets`  
`from IPython.display import display`

### **3\. Core Logic: Thermal Defect Analysis**

**Function: check\_planar\_inertial\_defect(Ia: float, Ib: float, Ic: float) \-\> dict**

> * **Logic:** Calculates \\Delta I \= I\_c \- I\_a \- I\_b.  
> * **Thresholding:** \* If strictly planar, \\Delta I should be 0\.  
  * If \\Delta I is small and *negative* (e.g., \-0.1 to \-1.0 amu·Å²), it is likely due to out-of-plane zero-point vibrational motion (thermal defect), not physical non-planarity.  
  * Emit an IPython HTML warning if a negative defect is detected, advising the user to lock out-of-plane coordinates.

### **4\. Core Logic: Symmetry UI**

**Class: SymmetryControllerUI**

> * **Logic:** Initializes a molsym.Molecule object from the COM-aligned coordinates.  
> * **UI Elements:**  
  * Iterates over all isomers.  
  * Renders a widgets.Dropdown populated with valid Schönflies symbols (C1, Cs, C2v, etc.), defaulting to the molsym detected point group.  
  * Binds an observe callback to the dropdown: if the user manually overrides (e.g., selecting Cs over C1), it updates a global symmetry\_override\_dict.