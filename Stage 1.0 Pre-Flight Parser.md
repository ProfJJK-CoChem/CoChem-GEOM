# **Stage 1.0 — Ingestion & Pre-Flight Syntax Validator**

**Target File:** cochem\_geom\_ingest\_parser.py

### **1\. Purpose & Scope**

Safely parse incoming experimental data (Pickett/SpycFit) and intercept malformed assignments before tensor generation begins. Actively filters isotopic assignments against natural abundance thresholds.

### **2\. Required Imports & Dependencies**

import re  
import json  
import pandas as pd  
from typing import Dict, List, Optional  
from dataclasses import dataclass

### **3\. Data Structures**

@dataclass  
class ExperimentalConstant:  
    species\_id: str  
    constant\_type: str \# e.g., 'A', 'B', 'C', 'DJ'  
    value\_mhz: float  
    uncertainty\_mhz: float  
    is\_isotopologue: bool  
    abundance\_percentage: float

### **4\. Core Class: SpectraIngestionEngine**

**Method: parse\_pickett\_par(self, par\_path: Path) \-\> List\[ExperimentalConstant\]**

> * **Logic:** Reads .par files line-by-line.  
> * **Regex Engine:** Must use strict regex r"^\\s\*(\\d+)\\s+(\\d+)\\s+(\[-\\.\\dEDe\]+)\\s+(\[-\\.\\dEDe\]+)" to trap the quantum parameter ID, value, and error.  
> * **Error Trap:** If parsing encounters a string-formatting shift (e.g., overlapping columns due to Fortran 1P,E15.8 truncation), it must raise a MalformedPickettError.

**Method: filter\_isotopic\_abundance(self, constants: List\[ExperimentalConstant\], threshold: float \= 0.01) \-\> List\[ExperimentalConstant\]**

> * **Logic:** Sweeps the ingested data. If an isotopologue represents \< 0.01\\% natural abundance (and the user hasn't flagged \--enriched), it must strip the constant from the fitting pool and log it to triage\_report.json as "suppressed\_due\_to\_noise".

### **5\. Native Integration**

> * Must also natively read CoChem-SpycFit JSON payloads, mapping them directly to the ExperimentalConstant dataclasses without invoking the regex engine.