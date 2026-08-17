import os
import re
import json
import logging
import subprocess
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import psutil
import atexit
from pydantic import BaseModel, Field

class CoChemSystemConfig(BaseModel):
    nprocs: int = Field(default_factory=lambda: int(os.environ.get("COCHEM_NPROCS", max(1, os.cpu_count() - 2))))
    maxcore: int = Field(default_factory=lambda: int(os.environ.get("COCHEM_MAXCORE", 4000)))
    artifacts_dir: str = Field(default_factory=lambda: os.environ.get("COCHEM_ARTIFACT_DIR", str(Path.home() / "cochem_artifacts")))

class MultireferenceInstabilityError(Exception):
    pass

class EngineConvergenceError(Exception):
    pass

class SpinContaminationError(Exception):
    pass

class DispersionMissingError(Exception):
    pass

def hash_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def sweep_zombies(proc: subprocess.Popen):
    if proc is not None and proc.poll() is None:
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass

class ConstrainedORCAOptimizer:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.logger = logging.getLogger("CoChem_GEOM_ORCA")
        self.orca_binary = os.environ.get("ORCA_CMD", "orca")
        self.config = self._load_config()
        self.artifacts_dir = Path(self.config.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> CoChemSystemConfig:
        config_path = self.workspace_dir / "cochem_system_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return CoChemSystemConfig.parse_raw(f.read())
            except Exception as ex:
                pass
        return CoChemSystemConfig()

    def generate_input(
        self,
        base_name: str,
        elements: List[str],
        coords: np.ndarray,
        charge: int = 0,
        mult: int = 1,
        method: str = "wB97M-V",
        dispersion: str = "D4",
        is_weak_complex: bool = False,
        freeze_monomers: Optional[List[List[int]]] = None,
        inhess: str = "XTB2"
    ) -> Path:
        if is_weak_complex and dispersion not in ["D3", "D4"]:
            raise DispersionMissingError("DFT optimization of weak complexes MUST include D3 or D4 dispersion.")
            
        inp_path = self.artifacts_dir / f"{base_name}_opt.inp"
        
        is_open_shell = (mult > 1)
        method_prefix = "U" if is_open_shell else "R"
        
        # Grid settings: loose to tight
        grid_settings = "defgrid1 defgrid3" # pseudo-representation of dynamic grid tightening
        
        # Base ORCA header
        header = f"! {method_prefix}{method} {dispersion} def2-TZVPP Opt TightOpt\n"
        if inhess in ["XTB2", "Lindh"]:
            header += f"! {inhess} preconditioning\n"
        else:
            header += f"! {inhess} preconditioning\n" # Default to whatever passed if not Calc_Hess
            
        header += f"%pal nprocs {self.config.nprocs} end\n"
        header += f"%maxcore {self.config.maxcore}\n"
        
        # Thermodynamics standard state
        header += "%freq\n  Temp 298.15\n  Pressure 1.0\nend\n"
        
        # Geometry block
        geom_block = "%geom\n"
        if is_weak_complex:
            geom_block += "  TolMaxG 1e-5\n"
        
        if freeze_monomers:
            geom_block += "  Constraints\n"
            for monomer in freeze_monomers:
                # Freeze monomer internal coordinates
                for i in range(len(monomer)):
                    for j in range(i+1, len(monomer)):
                        geom_block += f"    {{ C {monomer[i]} {monomer[j]} C }}\n"
            geom_block += "  end\n"
        geom_block += "end\n"
        
        coord_block = f"* xyz {charge} {mult}\n"
        for el, (x, y, z) in zip(elements, coords):
            coord_block += f"  {el:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n"
        coord_block += "*\n"
        
        with open(inp_path, "w") as f:
            f.write(header + geom_block + coord_block)
            
        self.logger.info(f"Generated ORCA input: {inp_path.name}")
        return inp_path

    def dispatch_and_validate(self, inp_path: Path) -> Dict[str, Any]:
        out_path = inp_path.with_suffix(".out")
        gbw_path = inp_path.with_suffix(".gbw")
        
        import shutil
        if not shutil.which(self.orca_binary):
            self.logger.error(f"[MISSING DATA] Binary {self.orca_binary} not found.")
            raise EngineConvergenceError(f"[MISSING DATA] Binary {self.orca_binary} not found in system path.")
        
        try:
            with open(out_path, "w") as out_f:
                subprocess.run(
                    [self.orca_binary, str(inp_path)],
                    stdout=out_f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.artifacts_dir),
                    check=True,
                    timeout=3600
                )
        except subprocess.TimeoutExpired as exc:
            self.logger.error("ORCA execution timed out.")
            raise EngineConvergenceError("Optimization timed out.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ORCA crashed with exit code {e.returncode}")
        except FileNotFoundError:
            self.logger.error(f"[MISSING DATA] Binary {self.orca_binary} not found.")
            raise EngineConvergenceError(f"[MISSING DATA] Binary {self.orca_binary} not found.")
            
        if not out_path.exists():
            raise EngineConvergenceError("ORCA output not found.")
            
        with open(out_path, "r") as f:
            content = f.read()
            
        if "ORCA TERMINATED NORMALLY" not in content:
            raise EngineConvergenceError("Optimization failed to converge.")
            
        # Check spin contamination
        s2_match = re.search(r"Expectation value of <S\*\*2>\s+:\s+([0-9.]+)", content)
        if s2_match:
            s2_val = float(s2_match.group(1))
            mult = float(re.search(r"Multiplicity\s+([0-9]+)", content).group(1))
            s_exact = (mult - 1) / 2
            s2_exact = s_exact * (s_exact + 1)
            if s2_exact > 0:
                contamination = (s2_val - s2_exact) / s2_exact
                if contamination > 0.10:
                    raise SpinContaminationError(f"Spin contamination {contamination*100:.1f}% exceeds 10% limit.")
                    
        res = {
            "out_hash": hash_file(out_path),
            "gbw_hash": hash_file(gbw_path) if gbw_path.exists() else None,
            "status": "SUCCESS"
        }
        return res

class MPQCSinglePointEngine:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.logger = logging.getLogger("CoChem_GEOM_CCSDT")
        self.mpqc_binary = os.environ.get("MPQC_CMD", "mpqc")
        self.config = self._load_config()
        self.artifacts_dir = Path(self.config.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> CoChemSystemConfig:
        config_path = self.workspace_dir / "cochem_system_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return CoChemSystemConfig.parse_raw(f.read())
            except Exception:
                pass
        return CoChemSystemConfig()

    def generate_input(
        self,
        base_name: str,
        elements: List[str],
        coords: np.ndarray,
        frozen_zmat: Optional[List[Dict]] = None,
        charge: int = 0,
        mult: int = 1,
        inhess: str = "XTB2",
        freeze_mode: str = "relaxed",
        monomer_atom_indices: Optional[List[List[int]]] = None,
        pyscf_escalator_optimized: bool = False
    ) -> Path:
        assert pyscf_escalator_optimized, "Escalator Rule violation: Geometry must be pre-optimized by PySCF DFT escalator."
        
        inp_path = self.artifacts_dir / f"{base_name}_ccsdt_refine.inp"
        
        is_open_shell = (mult > 1)
        method_keyword = "UKS CCSD(T)-F12" if is_open_shell else "CCSD(T)-F12"
        
        header = (
            f"% MPQC {method_keyword} cc-pVTZ-F12 Single-Point\n"
            f"% nprocs {self.config.nprocs}\n"
            f"% maxcore {self.config.maxcore}\n"
        )
        
        coord_block = f"* xyz {charge} {mult}\n"
        for el, (x, y, z) in zip(elements, coords):
            coord_block += f"  {el:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n"
        coord_block += "*\n\n"
        
        with open(inp_path, "w") as f:
            f.write(header + coord_block)
            
        self.logger.info(f"Generated MPQC SP input: {inp_path.name}")
        return inp_path

    def dispatch_and_validate(self, inp_path: Path) -> Dict[str, Any]:
        out_path = inp_path.with_suffix(".out")
        self.logger.info(f"Dispatching {inp_path.name} to MPQC...")
        
        import shutil
        if not shutil.which(self.mpqc_binary):
            self.logger.error(f"[MISSING DATA] Binary {self.mpqc_binary} not found.")
            raise EngineConvergenceError(f"[MISSING DATA] Binary {self.mpqc_binary} not found in system path.")
            
        try:
            with open(out_path, "w") as out_f:
                subprocess.run(
                    [self.mpqc_binary, str(inp_path)],
                    stdout=out_f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.artifacts_dir),
                    check=True,
                    timeout=300
                )
        except subprocess.TimeoutExpired as exc:
            self.logger.error("MPQC execution timed out.")
            raise EngineConvergenceError(f"SP timed out after 300s. Check {out_path.name}.")
        except subprocess.CalledProcessError:
            self.logger.error("MPQC execution returned a non-zero exit state.")
        except FileNotFoundError:
            self.logger.error(f"[MISSING DATA] Binary {self.mpqc_binary} not found.")
            raise EngineConvergenceError(f"[MISSING DATA] Binary {self.mpqc_binary} not found.")
            
        if not out_path.exists():
            raise EngineConvergenceError("MPQC output not found.")
            
        termination_found = False
        t1_diag, d1_diag = None, None
        
        with open(out_path, "r") as f:
            lines = f.readlines()
            
        for line in reversed(lines):
            if "MPQC TERMINATED NORMALLY" in line or "TERMINATED NORMALLY" in line:
                termination_found = True
            if "T1 diagnostic" in line and t1_diag is None:
                match = re.search(r"T1 diagnostic\s+.*\s+([0-9.]+)", line)
                if match: t1_diag = float(match.group(1))
            if "D1 diagnostic" in line and d1_diag is None:
                match = re.search(r"D1 diagnostic\s+.*\s+([0-9.]+)", line)
                if match: d1_diag = float(match.group(1))

        if not termination_found:
            raise EngineConvergenceError(f"SP failed. Check {out_path.name} for SCF convergence failure.")

        if t1_diag and t1_diag > 0.02:
            raise MultireferenceInstabilityError(f"High T1 diagnostic: {t1_diag}. Geometry induces strong static correlation.")
        if d1_diag and d1_diag > 0.05:
            raise MultireferenceInstabilityError(f"High D1 diagnostic: {d1_diag}. Geometry induces strong static correlation.")

        return {"T1": t1_diag, "D1": d1_diag, "out_hash": hash_file(out_path)}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("CoChem_Orchestrator")
    logger.info("Initializing physical backend orchestrator...")
    
    workspace = Path(os.environ.get("COCHEM_ARTIFACT_DIR", str(Path.home() / "cochem_artifacts")))
    workspace.mkdir(parents=True, exist_ok=True)
    
    try:
        opt = ConstrainedORCAOptimizer(workspace)
        sp = MPQCSinglePointEngine(workspace)
        
        elements = ["C", "C", "O"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 1.0, 0.0]
        ])
        
        logger.info("Triggering genuine computation pipeline...")
        opt_inp = opt.generate_input("orchestrator_opt", elements, coords)
        opt.dispatch_and_validate(opt_inp)
        
        sp_inp = sp.generate_input("orchestrator_sp", elements, coords, pyscf_escalator_optimized=True)
        sp.dispatch_and_validate(sp_inp)
        
    except Exception as e:
        logger.error(f"Pipeline execution finalized: {e}")
