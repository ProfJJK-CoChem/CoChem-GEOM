### **Template C — Workflow**

**Workflow goal** To map the definitive, context-safe execution sequence for CoChem-GEOM (v4.0). This workflow establishes the initialization schema, strict dual-ingestion parsers, interactive topological grouping matrices, the highly mathematical GPU-accelerated Z-matrix fitter, and the final rigorous publication reporting tier.  
**Stage map**

> * **Stage 0.0:** Master Setup & Headless Initialization (cochem\_geom\_setup.py)  
> * **Stage 1.0:** Ingestion & Pre-Flight Syntax Validator (cochem\_geom\_ingest\_parser.py)  
> * **Stage 1.1:** Isotope Math, Axis Detection & Fingerprinting (cochem\_geom\_ingest\_math.py)  
> * **Stage 2.1:** Symmetry Detection & Thermal Warnings (cochem\_geom\_eval\_sym.py)  
> * **Stage 2.2:** Variable Reduction & DoF Triage (cochem\_geom\_eval\_triage.py)  
> * **Stage 3.1:** Internal Coordinate & Quaternion Engine (cochem\_geom\_fitter\_core.py)  
> * **Stage 3.2:** GPU Multi-Seed Optimizer & MCMC Error Prop (cochem\_geom\_fitter\_optim.py)  
> * **Stage 3.3:** (Optional 3b+) Constrained CCSD(T) Refinement (cochem\_geom\_ccsdt\_refine.py)  
> * **Stage 4.0:** Correlation Filters & Publication Generator (cochem\_geom\_reporter\_latex.py)  
> * **Stage 4.1:** Interactive Output Visualization (cochem\_geom\_reporter\_ui.py)

**Detailed stage segments**  
**Stage 0.0: Master Setup & Headless Initialization**

> * **Purpose:** Initializes the air-gapped CoChem\_Artifacts workspace, connects to the core CoChem registry, and initializes concurrent data storage.  
> * **Inputs:** cochem\_system\_config.json, CLI arguments (for headless mode).  
> * **Outputs:** geom\_results.h5 (SWMR ready), references.bib, initialized logger.  
> * **Files created or modified:** cochem\_geom\_setup.py  
> * **Key dependencies:** h5py, json, pathlib, tqdm.  
> * **Key scientific or logic checks:** Verify HDF5 SWMR locking availability on the host filesystem.  
> * **Failure risks:** POSIX locks failing on network drives (NFS/SMB).  
> * **Suggested validation tests:** Trigger a parallel read to the .h5 file while writing to ensure SWMR stability.  
> * **Estimated coding size risk:** Low  
> * **Context safety note:** Kept highly isolated. Bootstraps the pipeline without loading heavy scientific libraries.

**Stage 1.0: Ingestion & Pre-Flight Syntax Validator**

> * **Purpose:** Safely parse incoming data and intercept malformed assignments before tensor generation.  
> * **Inputs:** Pickett files (.lin, .par, .fit), CoChem-SpycFit JSON payloads.  
> * **Outputs:** Cleaned Python dictionaries of transition frequencies and preliminary constants.  
> * **Files created or modified:** cochem\_geom\_ingest\_parser.py  
> * **Key dependencies:** re, json, pandas.  
> * **Key scientific or logic checks:** Syntax validation of quantum numbers (J, K\_a, K\_c); Isotope Natural Abundance Filter (flags \<0.01% species).  
> * **Failure risks:** Unorthodox formatting variants in user-uploaded legacy Pickett files.  
> * **Suggested validation tests:** Ingest a deliberately mangled .par file to confirm the regex intercepts it cleanly.  
> * **Estimated coding size risk:** Medium  
> * **Context safety note:** Splits ingestion from heavy math to protect context windows.

**Stage 1.1: Isotope Math, Axis Detection & Fingerprinting**

> * **Purpose:** Standardize the mass matrices and physically align the atomic coordinates.  
> * **Inputs:** Cleaned data dicts from Stage 1.0, base .xyz structures.  
> * **Outputs:** Center of Mass (COM) shifted tensors, Eckart aligned coordinates, geom\_provenance.json.  
> * **Files created or modified:** cochem\_geom\_ingest\_math.py  
> * **Key dependencies:** mendeleev, numpy, hashlib.  
> * **Key scientific or logic checks:** Exact monoisotopic mass querying; a \\leftrightarrow b \\leftrightarrow c axis reorientation detection.  
> * **Failure risks:** Eckart alignment divergence on near-spherical rotors.  
> * **Suggested validation tests:** Inject ^{13}\\text{C} substitution and verify mendeleev yields exactly 13.003354835 Da.  
> * **Estimated coding size risk:** Medium  
> * **Context safety note:** Handles pure NumPy matrix operations and exact float definitions.

**Stage 2.1: Symmetry Detection & Thermal Warnings**

> * **Purpose:** Define point-group symmetries and warn against vibration-induced planar distortion.  
> * **Inputs:** Eckart-aligned Cartesian matrices.  
> * **Outputs:** Detected symmetry lists, Planar Inertial Defect (\\Delta I) warning logs.  
> * **Files created or modified:** cochem\_geom\_eval\_sym.py  
> * **Key dependencies:** molsym, ipywidgets.  
> * **Key scientific or logic checks:** Computes thermal adjustment vectors to verify if apparent out-of-plane inertia is merely zero-point motion.  
> * **Failure risks:** Molsym numerical integration failure on noisy initial coordinates.  
> * **Suggested validation tests:** Manually force an override from C\_1 to C\_s and verify downstream internal coordinates update.  
> * **Estimated coding size risk:** Medium  
> * **Context safety note:** Separates GUI symmetry-override logic from the heavier matrix triage.

**Stage 2.2: Variable Reduction & DoF Triage**

> * **Purpose:** Establishes the optimizable variable set and determines the mathematical viability of the fit.  
> * **Inputs:** Verified symmetries, theoretical CCSD(T) offsets.  
> * **Outputs:** Frozen/Linked Parameter array, DoF Sufficiency Flag.  
> * **Files created or modified:** cochem\_geom\_eval\_triage.py  
> * **Key dependencies:** ipywidgets.  
> * **Key scientific or logic checks:** Strict execution gate: (\\text{Variables} \+ 1\) \\le \\text{Constants}.  
> * **Failure risks:** User stubbornly over-constraining the model resulting in 0 Float variables.  
> * **Suggested validation tests:** Attempt to float all atoms on a 3-isotope data set and ensure the system hard-locks with an "Unsufficient" warning.  
> * **Estimated coding size risk:** Medium  
> * **Context safety note:** Complex Drag-and-Drop parameter linking is isolated here.

**Stage 3.1: Internal Coordinate & Quaternion Engine**

> * **Purpose:** Maps the Cartesian space to stable Z-matrix forms for physically sound optimization.  
> * **Inputs:** DoF Parameter arrays, initial coordinates.  
> * **Outputs:** Live B-Matrix (Internal \\to Cartesian derivatives), Dynamic Bound Autotuner arrays.  
> * **Files created or modified:** cochem\_geom\_fitter\_core.py  
> * **Key dependencies:** scipy.spatial.transform (Rotation).  
> * **Key scientific or logic checks:** Bounds constrained by sum of mendeleev covalent radii \\pm 20\\%.  
> * **Failure risks:** Redundant internal coordinate generation leading to B-matrix singularities.  
> * **Suggested validation tests:** Trigger a linear angle 180^\\circ inversion and verify quaternion Eckart handling does not flip coordinates.  
> * **Estimated coding size risk:** High  
> * **Context safety note:** Intense mathematical abstraction. Must be coded completely independently.

**Stage 3.2: GPU Multi-Seed Optimizer & MCMC Error Prop**

> * **Purpose:** Executes the non-linear fitting cycles, guarantees global minimum convergence, and extracts precise uncertainties.  
> * **Inputs:** B-Matrix, Experimental Constants, Variable sets.  
> * **Outputs:** Converged parameters, Variance-Covariance matrix, r\_0/r\_s/r\_m/r\_e/r\_e^{\\text{SE}} structures to geom\_results.h5.  
> * **Files created or modified:** cochem\_geom\_fitter\_optim.py  
> * **Key dependencies:** torch / cupy, scipy.sparse.  
> * **Key scientific or logic checks:** Jacobian condition number check (\\text{cond}(J) \< 10^5); Analytical Kraitchman propagation.  
> * **Failure risks:** PyTorch VRAM out-of-memory on 100-atom dense Jacobians.  
> * **Suggested validation tests:** Force a highly correlated fit and verify the SVD pseudo-inverse kicks in to save the run.  
> * **Estimated coding size risk:** High  
> * **Context safety note:** This is the core physics engine. It utilizes sparse matrix logic where possible to defend memory.

**Stage 3.3: Constrained CCSD(T) Refinement (Stage 3b+)**

> * **Purpose:** Freezes newly determined experimental parameters and theoretically relaxes remaining coordinates.  
> * **Inputs:** Converged r\_e^{\\text{SE}} geometry.  
> * **Outputs:** orca.inp, orca.out, refined theoretical constants.  
> * **Files created or modified:** cochem\_geom\_ccsdt\_refine.py  
> * **Key dependencies:** subprocess, ORCA 6.1.1.  
> * **Key scientific or logic checks:** T\_1/D\_1 diagnostic checks to verify multireference stability of the new geometry.  
> * **Failure risks:** ORCA SCF divergence due to rigidly constrained geometric coordinates.  
> * **Suggested validation tests:** Check .inp to ensure %geom Constraints blocks are perfectly formatted.  
> * **Estimated coding size risk:** Medium  
> * **Context safety note:** Bridges directly with the external OpenMPI engine.

**Stage 4.0: Correlation Filters & Publication Generator**

> * **Purpose:** Transmute converged physics data into high-fidelity academic reports.  
> * **Inputs:** geom\_results.h5, geom\_provenance.json.  
> * **Outputs:** siunitx formatted .tex tables, references.bib, interactive correlation matrix threshold UI.  
> * **Files created or modified:** cochem\_geom\_reporter\_latex.py  
> * **Key dependencies:** ipywidgets.  
> * **Key scientific or logic checks:** Verifies assumed parameters have uncertainties replaced with Set Value.  
> * **Failure risks:** String concatenation errors producing malformed LaTeX syntax.  
> * **Suggested validation tests:** Compile the .tex output silently and verify return code 0\.  
> * **Estimated coding size risk:** Medium  
> * **Context safety note:** Dedicated purely to text manipulation and correlation slicing.

**Stage 4.1: Interactive Output Visualization**

> * **Purpose:** Build the visual telemetry loops (Heatmaps, 3D overlays, what-if sliders) for the final notebook cell.  
> * **Inputs:** Extracted parameters, residual errors.  
> * **Outputs:** Rendered Plotly and py3Dmol HTML widgets.  
> * **Files created or modified:** cochem\_geom\_reporter\_ui.py  
> * **Key dependencies:** plotly, py3Dmol.  
> * **Key scientific or logic checks:** Accurate mapping of displacement vectors onto the baseline 3D structure.  
> * **Failure risks:** Massive JS payload crashes notebook if rendering 100+ atom vibrational vectors at once.  
> * **Suggested validation tests:** Load a highly variant r\_0 vs r\_e frame and verify the vector arrows point to correct Cartesian shifts.  
> * **Estimated coding size risk:** Low  
> * **Context safety note:** Offloads rendering math directly to the client browser.

**Context risk points** Stages 3.1 and 3.2 are the most complex blocks in the ecosystem due to Z-matrix Jacobian derivations and GPU batching. They must be coded entirely alone.  
**Save-point recommendation** Please copy this text and save it externally as CoChem\_GEOM\_Workflow\_Plan\_v4.md to lock our execution roadmap.  
**Next safest segment to implement** Do you authorize proceeding into Coding Mode to build **Stage 0.0: Master Setup & Headless Initialization** (cochem\_geom\_setup.py) to construct the SWMR HDF5 wrapper, bibliography framework, and registry pipeline connections?