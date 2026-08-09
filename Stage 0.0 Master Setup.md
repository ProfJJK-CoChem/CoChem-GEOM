# **Stage 0.0 — Master Setup & Headless Initialization**

**Target File:** cochem\_geom\_setup.py

### **1\. Purpose & Scope**

To securely initialize the air-gapped CoChem\_Artifacts workspace, construct the Single-Writer Multiple-Reader (SWMR) HDF5 data framework, connect to the core CoChem hardware registry, and initialize the automated references.bib generator.

### **2\. Required Imports & Dependencies**

`import os`  
`import json`  
`import logging`  
`from pathlib import Path`  
`import h5py`  
`from tqdm import tqdm # For headless progress`

### **3\. Core Class: GeomEnvironmentOrchestrator**

**Initialization:**

> * Needs to locate the cochem\_system\_config.json via $HOME/CoChem\_Artifacts/Registry/.  
> * Must parse and store: cpu\_cores, system\_ram\_gb, cuda\_available.

**Method: \_initialize\_swmr\_lake(self, workspace\_name: str) \-\> Path**

> * **Logic:** Creates geom\_results.h5 inside $HOME/CoChem\_Artifacts/\[workspace\_name\]/.  
> * **Constraint:** Must instantiate with libver='latest' and enable SWMR mode (f.swmr\_mode \= True).  
> * **Validation:** Before exiting, it must attempt a parallel read-lock test. If POSIX locks fail (e.g., due to an NFS network drive), it must emit a warning and fall back to standard h5py locks.

**Method: \_initialize\_provenance(self, workspace\_name: str) \-\> None**

> * **Logic:** Creates an empty geom\_provenance.json dictionary.  
> * **Structure:** { "session\_id": UUID, "git\_commit": "", "hardware\_profile": {}, "citations": \[\] }  
> * Creates references.bib and seeds it with the base Pickett/Kraitchman/Watson DOIs.

### **4\. Headless Execution Sub-Layer**

> * If \_\_name\_\_ \== "\_\_main\_\_", argparse must accept \--config fit\_config.json and \--headless.  
> * If \--headless is True, bind the internal logger directly to tqdm to provide terminal output instead of attempting to draw ipywidgets.