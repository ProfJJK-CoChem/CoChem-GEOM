# **Stage 2.2 — Variable Reduction & DoF Triage**

**Target File:** cochem\_geom\_eval\_triage.py

### **1\. Purpose & Scope**

Establishes the final optimizable Z-matrix variable set through interactive parameter grouping, and determines the mathematical viability of the fit based on degrees of freedom.

### **2\. Required Dependencies**

`import numpy as np`  
`import ipywidgets as widgets`

### **3\. Core Matrix Setup**

**Class: VariableTriageEngine**

> * **Logic:** Translates the 3N Cartesian coordinates into 3N-6 non-redundant internal coordinates (Bonds, Angles, Dihedrals).  
> * **The "Hydrogen Lock":** Implements a fast-pass routine that scans atomic symbols. If symbol \== 'H', it sets is\_frozen \= True in the parameter matrix, significantly reducing the DoF.

**Method: apply\_theoretical\_offsets(self, primary\_idx: int, linked\_indices: list, ccsdt\_geometry: np.ndarray)**

> * **Logic:** For pseudo-symmetric rings (e.g., weakly distorted benzene), users can link multiple bonds to a single optimizable parameter. The engine locks the *differences* between these bonds based on the ccsdt\_geometry offsets. Only the primary\_idx floats; the others shift identically.

**Method: evaluate\_sufficiency(self, num\_constants: int, float\_variables: int) \-\> widgets.HTML**

> * **Logic:** \* margin \= num\_constants \- float\_variables  
  * If margin \>= 1: Return a Green HTML badge \[SUFFICIENT: \+X DoF\].  
  * If margin \< 1: Return a Red HTML badge \[UNSUFFICIENT: Need \+Y Constants or Freeze \+Y Vars\].  
  * *Gatekeeper:* If red, sets engine.is\_locked \= True, preventing execution of Stage 3\.