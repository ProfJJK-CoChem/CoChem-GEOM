import os
import glob
import ast
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def analyze_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    
    # 1. Hardcoded paths
    if "C:\\" in content or "D:\\" in content or "/home/" in content:
        issues.append("Hardcoded paths detected")
    
    # 2. Print statements vs logging
    if "print(" in content:
        issues.append("print() statements used (should use logging)")
    
    # 3. Subprocess safety
    if "subprocess.run" in content:
        if "check=True" not in content:
            issues.append("subprocess.run without check=True")
        if "timeout=" not in content:
            issues.append("subprocess.run without timeout")
        if "try:" not in content.split("subprocess.run")[0][-50:]:
            pass # hard to check statically, but we can look
    
    # 4. Pydantic Models missing
    if "pydantic" not in content.lower() and "class " in content and "def " in content:
        issues.append("Missing Pydantic for models/classes (potential)")

    # 5. Method Matrix Checks
    if "defgrid3" in content and "defgrid1" not in content:
        issues.append("Method Matrix Violation: Missing defgrid1 optimization loop")
    if "Grid3" in content or "Grid5" in content:
        issues.append("Method Matrix Violation: Deprecated Grid3/Grid5 terminology")
    if "Calc_Hess true" in content or "Calc_Hess True" in content:
        issues.append("Method Matrix Violation: Calc_Hess true used instead of InHess XTB2/Lindh")
    
    return issues

if __name__ == "__main__":
    files = glob.glob('**/*.py', recursive=True)
    for f in files:
        if 'tests/' in f or 'venv' in f:
            continue
        issues = analyze_file(f)
        if issues:
            logger.info(f"--- {f} ---")
            for i in set(issues):
                logger.info(f"  - {i}")
