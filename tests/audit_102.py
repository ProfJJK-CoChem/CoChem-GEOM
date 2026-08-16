import sys
import os

try:
    from cochem_geom.engine import hessian_theory
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

try:
    # Initialize TS optimization with step=0.1
    ts_opt = hessian_theory.TSOptimization(step=0.1)
    
    # Check if the step was corrected
    print(f"Executed with step: {ts_opt.step}")
    if hasattr(ts_opt, 'arxiv_doi'):
        print(f"ArXiv DOI: {ts_opt.arxiv_doi}")
    
    if ts_opt.step == 0.005:
        print("Audit passed: Step size was corrected to 0.005.")
    else:
        print("Audit failed: Step size was not corrected.")
except Exception as e:
    print(f"Execution Error: {e}")
    sys.exit(1)
