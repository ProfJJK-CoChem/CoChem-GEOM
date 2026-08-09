### **Architecture Mode: Final CoChem-GEOM (v4.0) Master Blueprint**

**Project Intent & Scope** **CoChem-GEOM** serves as the definitive precision molecular structure determination module for the CoChem ecosystem. It bridges the gap between theoretical quantum chemistry (ORCA 6.1.1, CoChem-BASE) and experimental rotational observables (CoChem-TORQ, CoChem-SpycFit). Designed to handle everything from highly fluxional 3-atom van der Waals complexes up to rigid 100-atom frameworks, it derives exact geometric parameters (r\_0, r\_s, r\_z, r\_m, r\_e, r\_e^{\\text{SE}}). The v4.0 architecture integrates all 28 approved deep, moderate, and shallow enhancements—featuring dynamic bounds autotuning, sparse matrix inversions, Pickett pre-flight syntax validation, interactive correlation filters, and comprehensive publication generators.

### **Hierarchical Module Architecture**

`CoChem_Artifacts/ (Air-Gapped Data & Workspace Tier)`  
   `├── 00-SETUP-AA   : cochem_geom_setup.py          (Environment, SWMR HDF5, Headless UI & BibTeX Init)`  
   `├── 01-INGEST-AA  : cochem_geom_ingest.py         (Pickett Validator, Isotope Filter & Fingerprinting)`  
   `├── 02-EVAL-AA    : cochem_geom_eval.py           (HITL Symmetry, Grouping & Thermal Defect Warnings)`  
   `├── 03-FIT-AA     : cochem_geom_fitter.py         (GPU Fitter, Autotuned Bounds, Sparse Hessians & MCMC)`  
   `├── 03b-REFINE-AA : cochem_geom_ccsdt_refine.py   (Constrained CCSD(T) ORCA Refinement)`  
   `└── 04-REPORT-AA  : cochem_geom_reporter.py       (siunitx Tables, Correlation Filters & Methods Generator)`

#### **Stage 0.0: Automated Setup & Headless Config (cochem\_geom\_setup.py)**

> * **Responsibilities:**  
  * Initializes the isolated workspace inside $HOME/CoChem\_Artifacts/\[workspace\_name\]/.  
  * Parses cochem\_system\_config.json to inspect CPU cores, system RAM, CUDA GPU availability, and micro-silo paths.  
  * **Single-Writer Multiple-Reader (SWMR) HDF5 Engine:** Initializes geom\_results.h5 with concurrency locks for live telemetry.  
  * **Headless Batch Execution & Status Bar:** Enables CLI execution (python \-m cochem\_geom.engine \--config fit\_config.json) featuring a lightweight tqdm terminal progress bar for high-throughput automated workflows.  
  * **Automated Citation Extractor:** Initializes references.bib to harvest DOIs for Pickett's SPFIT, Kraitchman/Watson models, and ORCA VPT2 literature automatically during execution.

#### **Stage 1.0: Master Ingestion & Pre-Flight Validation (cochem\_geom\_ingest.py)**

> * **Responsibilities:**  
  * **Dual Native Ingestion & Syntax Validator:** Parses native CoChem-SpycFit JSON/Parquet outputs alongside Pickett .lin/.fit/.par files. Implements a **Pre-Flight Parser** to intercept malformed quantum number assignments (J, K\_a, K\_c) before launching fits.  
  * **Automated Isotope Natural Abundance Filtering:** Automatically flags experimental rotational constants corresponding to rare isotopologues (natural abundance \< **0.01%**) for user review, preventing spectral noise from skewing fits unless enrichment is explicitly specified.  
  * **Mendeleev Monoisotopic Mass Engine:** Queries the mendeleev Python library for exact CIAAW/AME2020 monoisotopic masses.  
  * **Axis Reorientation Detection:** Tracks Eckart transformation matrices to automatically detect and correct a \\leftrightarrow b \\leftrightarrow c principal axis swaps upon isotopic substitution.  
  * **Cryptographic Data Fingerprinting:** Appends SHA-256 hashes of all input lists, ORCA logs, and atomic mass vectors.

#### **Stage 2a & 2b: Evaluation, Grouping & Integrity Checks (cochem\_geom\_eval.py)**

> * **Responsibilities:**  
  * **MolSym Point-Group & HITL Override:** Detects symmetry, rendering an interactive UI to allow users to force symmetry assignments (e.g., C\_1 \\rightarrow C\_s).  
  * **Inertial Defect Thermal Adjustment Warning:** Computes temperature dependence for planar inertial defects (\\Delta I \= I\_c \- I\_a \- I\_b) to alert users if out-of-plane zero-point vibrations are falsely emulating non-planar geometries.  
  * **Drag-and-Drop Parameter Grouping:** Provides an ipywidgets interface to visually group Z-matrix internal coordinates into "pseudo-symmetric" single-variable sets (e.g., linking all C-C bonds in a distorted aromatic ring utilizing CCSD(T) offsets).  
  * **VPT2 Resonance Alerts & Laurie Auto-Initialization:** Scans ORCA VPT2 outputs for Fermi/Coriolis resonances (\\Delta \\nu \< 15 \\text{ cm}^{-1}). Automatically initializes Laurie H/D bond-shortening parameters (\\delta r\_{\\text{HD}}) for r\_m fits using CCSD(T) harmonic frequencies.

#### **Stage 3.0: GPU Multi-Seed Fitter & Error Propagation (cochem\_geom\_fitter.py)**

> * **Responsibilities:**  
  * **Quaternion Eckart Transformations & Z-Matrix Engine:** Fits r\_0, r\_s, r\_z, r\_m, r\_e, r\_e^{\\text{SE}} structures using robust internal coordinates to prevent unphysical Cartesian distortions.  
  * **Dynamic Parameter Bounds Autotuning:** Automatically scales upper and lower bound constraints during non-linear optimization based on the sum of covalent radii from mendeleev, physically preventing atom overlap or bond dissociation.  
  * **Sparse Hessian Matrix Inversion:** Automatically triggers SciPy sparse matrix routines when calculating parameter covariance matrices for complexes exceeding **50 atoms**, drastically reducing RAM overhead.  
  * **Analytical Kraitchman Error Propagation:** Replaces empirical heuristics with full analytical Taylor-series error propagation of experimental moment-of-inertia uncertainties (\\sigma\_{\\Delta I}).  
  * **SVD Fallback & Structural Divergence Guard:** Aborts trial steps resulting in bonds \< **0.5 Å** or \> **5.0 Å**. Switches to SVD pseudo-inversion if the Jacobian condition number exceeds 10^5.  
  * **Multi-Seed Verification & MCMC Sampling:** Generates a randomized ensemble of seeds derived from theoretical offset magnitudes. Optionally executes Bayesian Markov Chain Monte Carlo (MCMC) sampling for corner probability density plots.

#### **Stage 3b+: Optional High-Level Refinement (cochem\_geom\_ccsdt\_refine.py)**

> * **Responsibilities:**  
  * Freezes newly fitted experimental geometries and executes an ORCA 6.1.1 constrained optimization on the remaining unconstrained parameters. Inherits grid and functional/Coupled-Cluster settings directly from the pipeline configuration.

#### **Stage 4.0: Publication & Interactive Reporting (cochem\_geom\_reporter.py)**

> * **Responsibilities:**  
  * **Interactive Correlation Matrix Threshold Filter:** In the Jupyter GUI, users can slide a threshold bar (e.g., r \> 0.85) to instantly highlight heavily correlated geometric parameter pairs in red, diagnosing ill-conditioned fits.  
  * **SiUnitx Publication Table Formatter:** Generates LaTeX tables properly aligned by decimal points using S\[table-format=...\] tags, ordered by energy isomer (horizontal) and atomic weight (vertical).  
  * **Methods & References Boilerplate Generator:** Automatically outputs a publication-ready methods section describing the fitting protocol, functional, error propagation, and the finalized references.bib file.  
  * **Diagnostic Overlays:** Renders Residual Waterfall Heatmaps via Plotly and 3D structural displacement vectors via py3Dmol.

### **Data Flow Architecture**

`[Experimental Data]                        [Theoretical Data]`  
 `Pickett (.lin/.fit/.par)                   ORCA VPT2 / CCSD(T)`  
 `CoChem-SpycFit Output                      CoChem-BASE / TORQ`  
         `│                                          │`  
         `└───────────────────┬──────────────────────┘`  
                             `▼`  
            `Stage 1.0: Ingestion & Pre-Flight`  
             `├── Pickett Syntax Validation & Isotope Filtering`  
             `├── Mendeleev Exact Isotopic Mass Lookup`  
             `└── SHA-256 Data Fingerprinting`  
                             `│`  
                             `▼`  
            `Stage 2a/2b: Evaluation & Grouping`  
             `├── HITL Symmetry & Drag-and-Drop Variables`  
             `├── Inertial Defect Thermal Warnings`  
             `└── DoF Validation: (Vars + 1 <= Constants)`  
                             `│`  
                             `▼`  
            `Stage 3.0: GPU Matrix Multi-Seed Fitter`  
             `├── Bounds Autotuning via Covalent Radii`  
             `├── Sparse Hessian Inversion (>50 Atoms)`  
             `├── Analytical Kraitchman & MCMC Sampling`  
             `└── Graceful Degradation (r_e^SE -> r_m -> r_0)`  
                             `│`  
              `┌──────────────┴──────────────┐`  
              `▼                             ▼`  
     `Stage 3b+: CCSD(T) Refine     Stage 4.0: Publication Engine`  
     `Constrained ORCA Execution      ├── siunitx LaTeX & BibTeX`  
     `Refined Assumed Parameters      ├── Pickett File Exporter`  
                                     `└── Interactive Correlation Filter`

### **Systemic Validation & Risk Mitigation**

> * **Syntax Mismatches:** Handled aggressively in Stage 1.0 by the Pickett pre-flight parser to prevent runtime crashes during matrix assignment.  
> * **Memory Exhaustion on \>50 Atom Systems:** Mitigated via SciPy sparse matrix inversions and GPU tensor chunking in Stage 3.0.  
> * **Ill-Conditioned Fits (Overfitting):** Checked mathematically via the Jacobian condition number (\<10^5) and visually identified via the Interactive Correlation Matrix Threshold Filter in Stage 4.0.

**Next Safest Action** The finalized CoChem-GEOM (v4.0) architecture is completely defined, addressing mathematical rigor, UX interactivity, memory safety, and structural publication requirements. The single safest next action is to transition to **Workflow Mode** to logically separate these concepts into safe, context-aware coding blocks.  
Do you authorize transitioning into **Workflow Mode** to map out the detailed stage segments (Stage 0.0 through Stage 4.0) for context-safe implementation?