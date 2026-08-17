import streamlit as st
import subprocess
import os
import sys
import psutil
import atexit
import logging
from pathlib import Path
from typing import Optional, List

# Core Directive 3: Replace print with logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CoChem_GEOM_Web")

st.set_page_config(page_title="CoChem-GEOM - Native Pipeline UI", layout="wide")

def kill_zombie_processes() -> None:
    """Core Directive 3: Graceful Failure & Subprocess Safety"""
    target_procs: List[str] = ['orca', 'xtb', 'mpi', 'crest']
    for proc in psutil.process_iter(['name']):
        try:
            # Null safety for proc.info dict
            name: str = (proc.info.get('name') or '').lower()
            if any(target in name for target in target_procs):
                logger.warning(f"Sweeping zombie process: {name} (PID: {proc.pid})")
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
atexit.register(kill_zombie_processes)

st.title("🔬 CoChem-GEOM Control Panel")
st.markdown("This UI executes raw, heavy mathematical payloads natively.")

with st.sidebar:
    st.header("Pipeline Configuration")
    target_smiles = st.text_input("Target SMILES", "CCO")
    run_mode = st.selectbox("Execution Mode", ["Fast", "Accurate"])

if st.button("🚀 Execute Default Pipeline"):
    with st.spinner(f"Triggering quantum physics executor for {target_smiles}..."):
        st.info("Initiating Physical Math Execution Pipeline...")
        
        module_dir: Path = Path(__file__).resolve().parent
        
        # Core Directive 1 & Rigorous Typing: Safely resolve artifact directory with consistent types
        env_artifact_dir: str = os.environ.get('COCHEM_ARTIFACT_DIR', str(Path.home() / 'cochem_artifacts'))
        artifact_dir: Path = Path(env_artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.copy()
        env["COCHEM_TARGET_H5"] = str(artifact_dir / "landscape.h5")
        env["COCHEM_TARGET_SMILES"] = target_smiles
        
        try:
            # Replaced no-op spoof with actual physical backend orchestrator target
            refine_script: Path = module_dir / "03b-REFINE-AA" / "cochem_geom_ccsdt_refine.py"
            
            if not refine_script.exists():
                st.error(f"Orchestrator not found at {refine_script}")
                st.stop()

            cmd: List[str] = [sys.executable, str(refine_script)]
            
            # Core Directive 3: Wrap subprocess.run in try/except with check=True and timeouts
            logger.info(f"Executing orchestrator: {cmd}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True, 
                timeout=3600, 
                cwd=str(module_dir),
                env=env
            )
            
            st.code(result.stdout[-3000:], language="text")
            if result.stderr:
                st.code(result.stderr[-3000:], language="text")
                
            st.success("✅ Execution Completed Natively. CPU load generated.")
                
        except subprocess.TimeoutExpired:
            logger.error("Execution timed out. Purging zombies.")
            st.error("Execution timed out. Purging zombies.")
            kill_zombie_processes()
        except subprocess.CalledProcessError as e:
            logger.error(f"Execution finished with non-zero exit code: {e.returncode}")
            st.warning(f"Execution finished with non-zero exit code: {e.returncode}")
            if e.stdout:
                st.code(e.stdout[-3000:], language="text")
            if e.stderr:
                st.error(e.stderr[-3000:])
            kill_zombie_processes()
        except Exception as e:
            logger.error(f"Pipeline crashed during physical execution: {str(e)}")
            st.error(f"Pipeline crashed during physical execution: {str(e)}")
            kill_zombie_processes()
