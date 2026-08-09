# CoChem-GEOM

**Precision Molecular Structure Determination Module**

CoChem-GEOM is the definitive precision molecular structure determination module for the CoChem ecosystem. It bridges the gap between theoretical quantum chemistry (ORCA 6.1.1, CoChem-BASE) and experimental rotational observables (CoChem-TORQ, CoChem-SpycFit). Designed to handle everything from highly fluxional 3-atom van der Waals complexes up to rigid 100-atom frameworks, it derives exact geometric parameters (r_0, r_s, r_z, r_m, r_e, r_e^{\\text{SE}}).

## 🏗️ Architecture

The CoChem-GEOM v4.0 architecture follows a strict stage-based execution flow:

### Stage 0.0: Automated Setup & Headless Configuration
- Initializes the isolated workspace inside $HOME/CoChem_Artifacts/[workspace_name]/.
- Parses cochem_system_config.json to inspect CPU cores, system RAM, CUDA GPU availability, and micro-silo paths.
- Single-Writer Multiple-Reader (SWMR) HDF5 Engine for concurrent data access.
- Headless batch execution with tqdm terminal progress bar.

### Stage 1.0: Master Ingestion & Pre-Flight Validation
- Dual native ingestion of Pickett .lin/.fit/.par files and CoChem-SpycFit JSON/Parquet outputs.
- Pre-flight parser to intercept malformed quantum number assignments (J, K_a, K_c).
- Automated isotope natural abundance filtering.

### Stage 2a & 2b: Evaluation, Grouping & Integrity Checks
- MolSym point-group detection with HITL override capability.
- Inertial defect thermal adjustment warnings.
- Drag-and-drop parameter grouping for Z-matrix internal coordinates.

### Stage 3.0: GPU Multi-Seed Fitter & Error Propagation
- Quaternion Eckart transformations and Z-matrix engine.
- Dynamic parameter bounds autotuning based on covalent radii.
- Sparse Hessian matrix inversion for large systems (>50 atoms).
- Analytical Kraitchman error propagation.

### Stage 3b+: Optional High-Level Refinement
- Constrained CCSD(T) ORCA refinement of fitted geometries.

### Stage 4.0: Publication & Interactive Reporting
- SiUnitx publication table formatter.
- Methods & references boilerplate generator.
- Diagnostic overlays with Residual Waterfall Heatmaps and 3D structural displacement vectors.

## 📦 Installation

CoChem-GEOM is installed as part of the CoChem ecosystem through the unified installer:

```bash
# Clone the main repository
git clone https://github.com/ProfJJK-CoChem/CoChem-BASE.git
cd CoChem-BASE

# Run the interactive installer
python -m interfaces.cochem_unity_installer_dashboard
```

## 🚀 Usage

### Headless Execution
```bash
python -m cochem_geom.engine --config fit_config.json
```

### Jupyter Integration
```python
from cochem_geom import setup, ingest, eval, fit, refine, report

# Initialize workspace
setup.initialize_workspace("my_molecule")

# Process data through pipeline
ingest.parse_data()
eval.evaluate_symmetry()
fit.optimize_geometry()
refine.refine_with_ccsdt()
report.generate_publication_tables()
```

## 📊 Data Flow

```
[Experimental Data]                        [Theoretical Data]
Pickett (.lin/.fit/.par)                   ORCA VPT2 / CCSD(T)
CoChem-SpycFit Output                      CoChem-BASE / TORQ
        │                                          │
        └───────────────────┬──────────────────────┘
                            ▼
           Stage 1.0: Ingestion & Pre-Flight
            ├── Pickett Syntax Validation & Isotope Filtering
            ├── Mendeleev Exact Isotopic Mass Lookup
            └── SHA-256 Data Fingerprinting
                            │
                            ▼
           Stage 2a/2b: Evaluation & Grouping
            ├── HITL Symmetry & Drag-and-Drop Variables
            ├── Inertial Defect Thermal Warnings
            └── DoF Validation: (Vars + 1 <= Constants)
                            │
                            ▼
           Stage 3.0: GPU Matrix Multi-Seed Fitter
            ├── Bounds Autotuning via Covalent Radii
            ├── Sparse Hessian Inversion (>50 Atoms)
            ├── Analytical Kraitchman & MCMC Sampling
            └── Graceful Degradation (r_e^SE -> r_m -> r_0)
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
    Stage 3b+: CCSD(T) Refine     Stage 4.0: Publication Engine
    Constrained ORCA Execution      ├── siunitx LaTeX & BibTeX
    Refined Assumed Parameters      ├── Pickett File Exporter
                                    └── Interactive Correlation Filter
```

## 📚 Key Features

- **GPU-Accelerated Fitting**: Leverages CUDA for high-performance optimization
- **Sparse Matrix Methods**: Efficient handling of large molecular systems (>50 atoms)
- **Multi-Seed Verification**: Robust global optimization with ensemble seeding
- **Analytical Error Propagation**: Full Taylor-series error propagation from experimental uncertainties
- **Interactive UI**: Jupyter-based visualization and parameter adjustment
- **Publication Ready**: Automated LaTeX table generation with proper siunitx formatting

## 🔧 Dependencies

CoChem-GEOM requires the following dependencies:
- Python 3.8+
- ORCA 6.1.1 (for CCSD(T) refinement)
- CUDA-compatible GPU (recommended for performance)
- CoChem-BASE ecosystem components

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contribution guidelines before submitting pull requests.

## 📬 Support

For support, please open an issue on the GitHub repository.