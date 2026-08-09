# **Stage 3.3 — Constrained CCSD(T) Refinement (Optional 3b+)**

**Target File:** cochem\_geom\_ccsdt\_refine.py

### **1\. Purpose & Scope**

Freeze newly determined experimental parameters and theoretically relax the remaining coordinates using ORCA 6.1.1.

### **2\. Required Imports & Dependencies**

`import os`  
`import subprocess`  
`import shutil`  
`from pathlib import Path`

### **3\. Engine Dispatch**

**Class: ConstrainedORCAOptimizer**

> * **Method: generate\_input(self, template\_args: dict, frozen\_zmat: list)**  
  * Injects the %geom Constraints block into the orca.inp file.  
  * Format: { C \[atom\_idx\_1\] \[atom\_idx\_2\] C } for bonds, { A ... } for angles.  
  * Adds \! TightOpt and \! DLPNO-CCSD(T) def2-TZVPP based on configuration.  
> * **Method: dispatch\_and\_validate(self)**  
  * Invokes the CoChem-CORE subprocess\_broker.py to handle MPI safety.  
  * Scans orca.out upon completion for T1 diagnostic and D1 diagnostic.  
  * If T1 \> 0.02 or D1 \> 0.05, raises MultireferenceInstabilityError, meaning the rigid coordinate locks have broken the single-reference assumption of Coupled Cluster theory.