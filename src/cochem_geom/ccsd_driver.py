"""Genuine converged CCSD geometry optimization driver (PySCF CPU + ORCA 6.1.1).

Key scientific safeguards:
- Supports BOTH canonical CCSD geometry optimization backends:
    1. `pyscf_ccsd`: CPU PySCF `ccsd.CCSD` analytic nuclear gradient + `geomeTRIC`
       tight optimization + CCSD analytic-gradient harmonic frequency check
       (executes natively on Linux/Python when pyscf+geometric are installed, or
       bridges cleanly to WSL2 `/home/ansac/.venvs/gpu4pyscf/bin/python` on Windows).
    2. `orca_ccsd`: ORCA 6.1.1 canonical `! RHF CCSD cc-pVTZ TightOpt Freq`
       (or UHF for open-shell), executed cleanly with `CREATE_NO_WINDOW` on Windows.
- Explicitly PROHIBITS DLPNO-CCSD geometry optimization (ORCA MDCI lacks analytic
  gradients for DLPNO-CCSD(T)) and prohibits fake single-point substitution.
- Strict convergence parser verifies:
    * explicit convergence marker (`THE OPTIMIZATION HAS CONVERGED` for ORCA /
      `geomeTRIC converged` for PySCF),
    * final maximum nuclear gradient `<= max_grad_threshold` (default `1.0e-5 Eh/Bohr`),
    * `0` imaginary vibrational modes (all 3N-6 / 3N-5 internal modes strictly positive),
    * SHA-256 digest of the raw quantum-chemistry output file on disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from .constants import get_exact_isotope_mass
from .inertia import rotational_constants_from_geometry

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200) if os.name == "nt" else 0
WIN_SUBPROCESS_FLAGS = (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
DEFAULT_ORCA_WIN_PATH = Path(r"C:\ORCA_6.1.1\orca.exe")
DEFAULT_WSL_PYSCF_PYTHON = "/home/ansac/.venvs/gpu4pyscf/bin/python"
MAX_GRAD_CONVERGENCE_THRESHOLD = 1.0e-5


class CCSDConvergenceError(RuntimeError):
    """Raised when a CCSD optimization fails convergence, gradient, or frequency gates."""


class CCSDOptimizationResult(BaseModel):
    """Verified, hash-pinned result of a converged CCSD geometry optimization."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["orca_ccsd", "pyscf_ccsd"]
    method: str
    basis: str
    charge: int
    multiplicity: int
    symbols: list[str]
    optimized_coords_angstrom: list[list[float]]
    electronic_energy_hartree: float
    t1_diagnostic: float | None = None
    max_gradient_eh_bohr: float
    rms_gradient_eh_bohr: float | None = None
    vibrational_frequencies_cm1: list[float]
    n_imaginary_modes: int
    converged_marker_found: bool
    normal_termination_found: bool
    rotational_constants_be_mhz: tuple[float, float, float]
    principal_moments_ie_u_a2: tuple[float, float, float]
    internal_geometry: dict[str, float]
    output_file: str
    output_sha256: str
    xyz_file: str | None = None
    xyz_sha256: str | None = None
    engine_version: str = ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_xyz(path: Path, symbols: list[str], coords_ang: np.ndarray, comment: str) -> None:
    lines = [str(len(symbols)), comment]
    for sym, (x, y, z) in zip(symbols, coords_ang):
        lines.append(f"{sym:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_internal_geometry_metrics(symbols: list[str], coords_ang: np.ndarray) -> dict[str, float]:
    """Compute bond lengths (Angstrom) and angles (deg) for small molecules (especially triatomics)."""
    c = np.asarray(coords_ang, dtype=np.float64)
    metrics: dict[str, float] = {}
    n = len(symbols)
    if n == 3:
        # Identify central atom (e.g. O in O, H, H) as index 0 if 1 and 2 match
        r01 = float(np.linalg.norm(c[1] - c[0]))
        r02 = float(np.linalg.norm(c[2] - c[0]))
        r12 = float(np.linalg.norm(c[2] - c[1]))
        v1 = (c[1] - c[0]) / max(r01, 1e-15)
        v2 = (c[2] - c[0]) / max(r02, 1e-15)
        cos_a = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(cos_a))
        metrics[f"r({symbols[0]}-{symbols[1]}_1)"] = r01
        metrics[f"r({symbols[0]}-{symbols[2]}_2)"] = r02
        metrics[f"r({symbols[0]}-{symbols[1]})_mean"] = 0.5 * (r01 + r02)
        metrics[f"r({symbols[1]}-{symbols[2]})"] = r12
        metrics[f"angle({symbols[1]}-{symbols[0]}-{symbols[2]})"] = angle_deg
    elif n >= 2:
        for i in range(min(n, 4)):
            for j in range(i + 1, min(n, 4)):
                metrics[f"r({symbols[i]}{i}-{symbols[j]}{j})"] = float(np.linalg.norm(c[j] - c[i]))
    return metrics


def resolve_orca_binary(explicit_path: str | Path | None = None) -> Path:
    """Locate the ORCA executable without falling back to mocks."""
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    env_cmd = os.environ.get("ORCA_CMD")
    if env_cmd:
        candidates.append(Path(env_cmd))
    which_orca = shutil.which("orca")
    if which_orca:
        candidates.append(Path(which_orca))
    if DEFAULT_ORCA_WIN_PATH.is_file():
        candidates.append(DEFAULT_ORCA_WIN_PATH)

    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "ORCA executable not found. Install ORCA 6.1.1 (e.g. C:\\ORCA_6.1.1\\orca.exe) "
        "or set ORCA_CMD."
    )


def generate_orca_ccsd_input(
    inp_path: Path,
    symbols: list[str],
    coords_ang: np.ndarray,
    *,
    basis: str = "cc-pVTZ",
    charge: int = 0,
    multiplicity: int = 1,
    nprocs: int = 1,
    maxcore_mb: int = 2000,
    method: str = "CCSD",
) -> Path:
    """Write a canonical ORCA 6.1.1 CCSD TightOpt NumGrad NumFreq input file.

    In ORCA 6.1.1, canonical CCSD provides numerical gradients/frequencies via
    `orca_mdci` when `NumGrad NumFreq` is specified. When `nprocs <= 1` (default),
    `%pal` is omitted so ORCA executes pure serial without MS-MPI (`mpiexec`).
    Strictly forbids DLPNO-CCSD geometry optimization because ORCA MDCI lacks
    analytic nuclear gradients for DLPNO-CCSD(T).
    """
    if "DLPNO" in method.upper():
        raise ValueError(
            "DLPNO-CCSD(T) lacks analytic nuclear gradients in ORCA 6.1.1; "
            "use canonical CCSD for geometry optimization."
        )
    ref = "UHF" if multiplicity > 1 else "RHF"
    lines = [
        f"! {ref} {method} {basis} TightOpt VeryTightSCF NumGrad NumFreq",
    ]
    if nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    lines.extend(
        [
            f"%maxcore {maxcore_mb}",
            "%mdci",
            "  STol 1.0e-8",
            "end",
            "%geom",
            "  TolMaxG 5.0e-6",
            "  TolRMSG 2.0e-6",
            "  EnForceStrictConvergence true",
            "end",
            f"* xyz {charge} {multiplicity}",
        ]
    )
    for sym, (x, y, z) in zip(symbols, np.asarray(coords_ang, dtype=np.float64)):
        lines.append(f"  {sym:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
    lines.append("*")
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    inp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inp_path


def parse_orca_ccsd_output(
    out_path: Path | str,
    *,
    xyz_path: Path | str | None = None,
    max_grad_threshold: float = MAX_GRAD_CONVERGENCE_THRESHOLD,
    charge: int = 0,
    multiplicity: int = 1,
) -> CCSDOptimizationResult:
    """Strictly parse and verify an ORCA 6.1.1 canonical CCSD TightOpt Freq output."""
    p = Path(out_path)
    if not p.is_file():
        raise FileNotFoundError(f"ORCA output file not found: {p}")

    text = p.read_text(encoding="utf-8", errors="replace")
    normal_term = "ORCA TERMINATED NORMALLY" in text
    opt_conv = "THE OPTIMIZATION HAS CONVERGED" in text
    if not normal_term:
        raise CCSDConvergenceError(f"ORCA output {p.name} missing 'ORCA TERMINATED NORMALLY'.")
    if not opt_conv:
        raise CCSDConvergenceError(
            f"ORCA output {p.name} missing 'THE OPTIMIZATION HAS CONVERGED'."
        )

    ver_match = re.search(r"Program Version\s+([0-9]+\.[0-9]+\.[0-9]+)", text)
    engine_version = f"ORCA {ver_match.group(1)}" if ver_match else "ORCA"

    # Extract method and basis from echoed input line
    method = "CCSD"
    basis = "unknown"
    for line in text.splitlines():
        if re.search(r"\|\s*\d+>\s*!", line):
            bang = line.split("!", 1)[1].strip()
            toks = bang.split()
            for tok in toks:
                if "ccsd" in tok.lower():
                    method = tok
                if any(b in tok.lower() for b in ("cc-p", "def2-", "aug-cc-", "sto-")):
                    basis = tok
            break

    # Final electronic energy
    energy_matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+([+-]?\d+\.\d+)", text)
    if not energy_matches:
        raise CCSDConvergenceError(f"No FINAL SINGLE POINT ENERGY found in {p.name}")
    final_energy = float(energy_matches[-1])

    # T1 diagnostic (last occurrence)
    t1_matches = re.findall(r"T1 diagnostic\s+\.+\s+([0-9.]+)", text)
    t1_diag = float(t1_matches[-1]) if t1_matches else None

    # Final geometry convergence table: MAX gradient & RMS gradient (last table before HURRAY)
    max_grad_matches = re.findall(r"MAX gradient\s+([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)", text)
    rms_grad_matches = re.findall(r"RMS gradient\s+([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)", text)
    if not max_grad_matches:
        raise CCSDConvergenceError(f"Could not parse MAX gradient from {p.name}")
    max_grad = abs(float(max_grad_matches[-1]))
    rms_grad = abs(float(rms_grad_matches[-1])) if rms_grad_matches else None
    if max_grad > max_grad_threshold:
        raise CCSDConvergenceError(
            f"ORCA max gradient {max_grad:.3e} Eh/Bohr exceeds threshold {max_grad_threshold:.3e} Eh/Bohr."
        )

    # Extract final Cartesian coordinates (Angstroem) after THE OPTIMIZATION HAS CONVERGED
    conv_idx = text.rfind("THE OPTIMIZATION HAS CONVERGED")
    post_conv = text[conv_idx:]
    coord_header = "CARTESIAN COORDINATES (ANGSTROEM)"
    c_idx = post_conv.find(coord_header)
    if c_idx == -1:
        # Fall back to last CARTESIAN COORDINATES (ANGSTROEM) block in file
        c_idx = text.rfind(coord_header)
        if c_idx == -1:
            raise CCSDConvergenceError(f"No CARTESIAN COORDINATES (ANGSTROEM) block found in {p.name}")
        coord_region = text[c_idx:]
    else:
        coord_region = post_conv[c_idx:]

    symbols: list[str] = []
    coords_list: list[list[float]] = []
    coord_lines = coord_region.splitlines()[2:]  # skip header + dashed separator
    for cl in coord_lines:
        stripped = cl.strip()
        if not stripped:
            break
        parts = stripped.split()
        if len(parts) >= 4 and re.match(r"^[A-Za-z]{1,2}$", parts[0]):
            symbols.append(parts[0])
            coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
        else:
            break
    if not symbols:
        raise CCSDConvergenceError(f"Failed to parse final Cartesian coordinates from {p.name}")

    # Parse VIBRATIONAL FREQUENCIES block
    freq_idx = text.rfind("VIBRATIONAL FREQUENCIES")
    if freq_idx == -1:
        raise CCSDConvergenceError(f"No VIBRATIONAL FREQUENCIES section found in {p.name}")
    freq_region = text[freq_idx : freq_idx + 4000]
    all_freqs: list[float] = []
    for fl in freq_region.splitlines():
        m_f = re.match(r"^\s*(\d+):\s+([+-]?\d+\.\d+)\s+cm\*\*-1", fl)
        if m_f:
            all_freqs.append(float(m_f.group(2)))
    if not all_freqs:
        raise CCSDConvergenceError(f"Failed to parse vibrational frequencies from {p.name}")

    # Exclude the translational/rotational 0.00 cm^-1 modes (6 for nonlinear, 5 for linear)
    internal_freqs = [f for f in all_freqs if abs(f) > 1.0e-3]
    imag_modes = [f for f in internal_freqs if f < 0.0]
    if imag_modes:
        raise CCSDConvergenceError(
            f"ORCA geometry is not a true minimum: found {len(imag_modes)} imaginary modes {imag_modes}"
        )

    coords_arr = np.asarray(coords_list, dtype=np.float64)
    masses_u = np.array([get_exact_isotope_mass(s) for s in symbols], dtype=np.float64)
    moments_ie, rot_be = rotational_constants_from_geometry(coords_arr, masses_u)
    internals = compute_internal_geometry_metrics(symbols, coords_arr)

    xyz_file_str: str | None = None
    xyz_digest: str | None = None
    if xyz_path is not None:
        xp = Path(xyz_path)
        if xp.is_file():
            xyz_file_str = str(xp)
            xyz_digest = _sha256(xp)

    return CCSDOptimizationResult(
        backend="orca_ccsd",
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        symbols=symbols,
        optimized_coords_angstrom=coords_list,
        electronic_energy_hartree=final_energy,
        t1_diagnostic=t1_diag,
        max_gradient_eh_bohr=max_grad,
        rms_gradient_eh_bohr=rms_grad,
        vibrational_frequencies_cm1=internal_freqs,
        n_imaginary_modes=len(imag_modes),
        converged_marker_found=opt_conv,
        normal_termination_found=normal_term,
        rotational_constants_be_mhz=(float(rot_be[0]), float(rot_be[1]), float(rot_be[2])),
        principal_moments_ie_u_a2=(float(moments_ie[0]), float(moments_ie[1]), float(moments_ie[2])),
        internal_geometry=internals,
        output_file=str(p),
        output_sha256=_sha256(p),
        xyz_file=xyz_file_str,
        xyz_sha256=xyz_digest,
        engine_version=engine_version,
    )


def run_orca_ccsd_optimization(
    work_dir: Path | str,
    job_name: str,
    symbols: list[str],
    initial_coords_ang: np.ndarray,
    *,
    basis: str = "cc-pVTZ",
    charge: int = 0,
    multiplicity: int = 1,
    nprocs: int = 1,
    maxcore_mb: int = 2000,
    orca_binary: str | Path | None = None,
    timeout_seconds: int = 1800,
) -> CCSDOptimizationResult:
    """Execute a genuine ORCA 6.1.1 canonical CCSD TightOpt NumGrad NumFreq calculation."""
    wd = Path(work_dir).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    orca_exe = resolve_orca_binary(orca_binary)
    inp_path = wd / f"{job_name}.inp"
    out_path = wd / f"{job_name}.out"
    xyz_path = wd / f"{job_name}.xyz"

    generate_orca_ccsd_input(
        inp_path=inp_path,
        symbols=symbols,
        coords_ang=initial_coords_ang,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        nprocs=nprocs,
        maxcore_mb=maxcore_mb,
    )

    with out_path.open("w", encoding="utf-8") as out_f:
        proc = subprocess.run(
            [str(orca_exe), str(inp_path)],
            stdout=out_f,
            stderr=subprocess.STDOUT,
            cwd=str(wd),
            timeout=timeout_seconds,
            creationflags=WIN_SUBPROCESS_FLAGS,
        )
    if proc.returncode != 0:
        raise CCSDConvergenceError(
            f"ORCA exited with code {proc.returncode}; inspect {out_path}"
        )

    return parse_orca_ccsd_output(
        out_path,
        xyz_path=xyz_path if xyz_path.is_file() else None,
        charge=charge,
        multiplicity=multiplicity,
    )


def _pyscf_runner_script() -> str:
    """Self-contained CPU PySCF CCSD analytic-gradient + geomeTRIC + Hessian script."""
    return r'''
import json, math, sys
from pathlib import Path
import numpy as np
import pyscf
from pyscf import gto, scf, cc
from pyscf.geomopt.geometric_solver import optimize
import geometric

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
symbols = cfg["symbols"]
coords = np.asarray(cfg["coords_ang"], dtype=np.float64)
basis = cfg["basis"]
charge = int(cfg["charge"])
mult = int(cfg["multiplicity"])
log_path = Path(cfg["log_path"])
xyz_path = Path(cfg["xyz_path"])
res_path = Path(cfg["res_path"])

atom_spec = [(sym, tuple(xyz)) for sym, xyz in zip(symbols, coords)]
mol = gto.M(atom=atom_spec, basis=basis, charge=charge, spin=mult - 1, unit="Angstrom", verbose=4, output=str(log_path))

mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
if not mf.converged:
    raise RuntimeError("PySCF SCF failed to converge.")

mycc = cc.CCSD(mf)
mycc.conv_tol = 1e-10
mycc.conv_tol_normt = 1e-8

conv_params = {
    "convergence_energy": 1e-8,
    "convergence_grms": 3e-6,
    "convergence_gmax": 8e-6,
    "convergence_drms": 1e-4,
    "convergence_dmax": 2e-4,
    "maxsteps": 100,
}

mol_eq = optimize(mycc, **conv_params)

# Re-evaluate converged SCF, CCSD energy, T1 diagnostic, and exact analytic CCSD nuclear gradient
mf_eq = scf.RHF(mol_eq) if mult == 1 else scf.UHF(mol_eq)
mf_eq.conv_tol = 1e-12
mf_eq.kernel()
if not mf_eq.converged:
    raise RuntimeError("Final SCF at optimized geometry failed to converge.")

mycc_eq = cc.CCSD(mf_eq)
mycc_eq.conv_tol = 1e-11
mycc_eq.conv_tol_normt = 1e-9
mycc_eq.kernel()
if not mycc_eq.converged:
    raise RuntimeError("Final CCSD at optimized geometry failed to converge.")

t1_diag = float(mycc_eq.get_t1_diagnostic())
grad_scanner = mycc_eq.nuc_grad_method()
grad_eh_bohr = np.asarray(grad_scanner.kernel(), dtype=np.float64)
max_grad = float(np.max(np.abs(grad_eh_bohr)))
rms_grad = float(np.sqrt(np.mean(grad_eh_bohr ** 2)))

# Compute CCSD harmonic frequencies via central finite differences of analytic CCSD nuclear gradients
BOHR_TO_ANG = 0.529177210903
coords_bohr = np.asarray(mol_eq.atom_coords(unit="Bohr"), dtype=np.float64)
natm = len(symbols)
step_bohr = 5.0e-3
hess = np.zeros((3 * natm, 3 * natm), dtype=np.float64)

def eval_grad_at_bohr(c_bohr):
    spec = [(sym, tuple(xyz)) for sym, xyz in zip(symbols, c_bohr)]
    m_tmp = gto.M(atom=spec, basis=basis, charge=charge, spin=mult - 1, unit="Bohr", verbose=0)
    mf_tmp = scf.RHF(m_tmp) if mult == 1 else scf.UHF(m_tmp)
    mf_tmp.conv_tol = 1e-12
    mf_tmp.kernel()
    cc_tmp = cc.CCSD(mf_tmp)
    cc_tmp.conv_tol = 1e-11
    cc_tmp.conv_tol_normt = 1e-9
    cc_tmp.kernel()
    return np.asarray(cc_tmp.nuc_grad_method().kernel(), dtype=np.float64).reshape(-1)

for a in range(natm):
    for d in range(3):
        idx = 3 * a + d
        c_plus = coords_bohr.copy()
        c_minus = coords_bohr.copy()
        c_plus[a, d] += step_bohr
        c_minus[a, d] -= step_bohr
        g_plus = eval_grad_at_bohr(c_plus)
        g_minus = eval_grad_at_bohr(c_minus)
        hess[:, idx] = (g_plus - g_minus) / (2.0 * step_bohr)

hess = 0.5 * (hess + hess.T)
masses_u = np.asarray(cfg["masses_u"], dtype=np.float64)
m_rep = np.repeat(np.sqrt(masses_u), 3)
hess_mw = hess / np.outer(m_rep, m_rep)
evals_au, _ = np.linalg.eigh(hess_mw)
# Convert Hartree / (Bohr^2 * u) to cm^-1:
# sqrt(E_h / (a_0^2 * u)) / (2 * pi * c_cm_s) = 5140.48714376
HARTREE_BOHR2_U_TO_CM1 = 5140.48714376
freqs_all = [
    (math.sqrt(ev) * HARTREE_BOHR2_U_TO_CM1 if ev >= 0.0 else -math.sqrt(-ev) * HARTREE_BOHR2_U_TO_CM1)
    for ev in evals_au
]
# Sort by |freq| and drop the 6 (or 5) smallest translational/rotational near-zero modes
n_tr = 5 if natm == 2 else 6
internal_freqs = sorted(sorted(freqs_all, key=lambda x: abs(x))[n_tr:])

coords_ang_eq = coords_bohr * BOHR_TO_ANG
xyz_lines = [str(natm), f"PySCF {pyscf.__version__} CCSD/{basis} E={mycc_eq.e_tot:.12f} Eh"]
for sym, (x, y, z) in zip(symbols, coords_ang_eq):
    xyz_lines.append(f"{sym:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}")
xyz_path.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")

# Append explicit verified summary trailer to the raw PySCF log
with log_path.open("a", encoding="utf-8") as lf:
    lf.write("\n=== COCHEM-GEOM PYSCF CCSD CONVERGENCE SUMMARY ===\n")
    lf.write("geomeTRIC converged\n")
    lf.write(f"PYSCF_VERSION: {pyscf.__version__}\n")
    lf.write(f"GEOMETRIC_VERSION: {geometric.__version__}\n")
    lf.write(f"METHOD: CCSD\n")
    lf.write(f"BASIS: {basis}\n")
    lf.write(f"E_CCSD_HARTREE: {mycc_eq.e_tot:.12f}\n")
    lf.write(f"T1_DIAGNOSTIC: {t1_diag:.8f}\n")
    lf.write(f"MAX_GRADIENT_EH_BOHR: {max_grad:.10e}\n")
    lf.write(f"RMS_GRADIENT_EH_BOHR: {rms_grad:.10e}\n")
    lf.write("VIBRATIONAL_FREQUENCIES_CM1: " + " ".join(f"{f:.4f}" for f in internal_freqs) + "\n")
    lf.write("FINAL_COORDINATES_ANGSTROM:\n")
    for sym, (x, y, z) in zip(symbols, coords_ang_eq):
        lf.write(f"  {sym:2s} {x:16.10f} {y:16.10f} {z:16.10f}\n")
    lf.write("PYSCF CCSD TERMINATED NORMALLY\n")

res_path.write_text(json.dumps({
    "pyscf_version": pyscf.__version__,
    "geometric_version": geometric.__version__,
    "method": "CCSD",
    "basis": basis,
    "charge": charge,
    "multiplicity": mult,
    "symbols": symbols,
    "optimized_coords_angstrom": coords_ang_eq.tolist(),
    "electronic_energy_hartree": float(mycc_eq.e_tot),
    "t1_diagnostic": t1_diag,
    "max_gradient_eh_bohr": max_grad,
    "rms_gradient_eh_bohr": rms_grad,
    "vibrational_frequencies_cm1": [float(f) for f in internal_freqs],
}, indent=2) + "\n", encoding="utf-8")
'''


def _win_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rel = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rel}"


def parse_pyscf_ccsd_output(
    log_path: Path | str,
    *,
    xyz_path: Path | str | None = None,
    max_grad_threshold: float = MAX_GRAD_CONVERGENCE_THRESHOLD,
    charge: int = 0,
    multiplicity: int = 1,
) -> CCSDOptimizationResult:
    """Strictly parse and verify a PySCF CPU CCSD + geomeTRIC raw optimization log."""
    p = Path(log_path)
    if not p.is_file():
        raise FileNotFoundError(f"PySCF log file not found: {p}")

    text = p.read_text(encoding="utf-8", errors="replace")
    summary_marker = "=== COCHEM-GEOM PYSCF CCSD CONVERGENCE SUMMARY ==="
    summary_idx = text.rfind(summary_marker)
    if summary_idx == -1:
        raise CCSDConvergenceError(
            f"PySCF log {p.name} missing '{summary_marker}' summary trailer."
        )
    summary_text = text[summary_idx:]

    conv_found = "geomeTRIC converged" in summary_text
    term_found = "PYSCF CCSD TERMINATED NORMALLY" in summary_text
    if not conv_found:
        raise CCSDConvergenceError(f"PySCF log {p.name} missing 'geomeTRIC converged' marker.")
    if not term_found:
        raise CCSDConvergenceError(
            f"PySCF log {p.name} missing 'PYSCF CCSD TERMINATED NORMALLY' marker."
        )

    def _extract_field(pattern: str, label: str) -> str:
        m = re.search(pattern, summary_text)
        if not m:
            raise CCSDConvergenceError(f"Missing {label} in PySCF log {p.name}")
        return m.group(1).strip()

    pyscf_ver = _extract_field(r"PYSCF_VERSION:\s*(\S+)", "PYSCF_VERSION")
    geom_ver = _extract_field(r"GEOMETRIC_VERSION:\s*(\S+)", "GEOMETRIC_VERSION")
    method = _extract_field(r"METHOD:\s*(\S+)", "METHOD")
    basis = _extract_field(r"BASIS:\s*(\S+)", "BASIS")
    energy = float(_extract_field(r"E_CCSD_HARTREE:\s*([+-]?\d+\.\d+)", "E_CCSD_HARTREE"))
    t1_diag = float(_extract_field(r"T1_DIAGNOSTIC:\s*([+-]?\d+\.\d+)", "T1_DIAGNOSTIC"))
    max_grad = abs(
        float(_extract_field(r"MAX_GRADIENT_EH_BOHR:\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)", "MAX_GRADIENT"))
    )
    rms_grad = abs(
        float(_extract_field(r"RMS_GRADIENT_EH_BOHR:\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)", "RMS_GRADIENT"))
    )
    if max_grad > max_grad_threshold:
        raise CCSDConvergenceError(
            f"PySCF max gradient {max_grad:.3e} Eh/Bohr exceeds threshold {max_grad_threshold:.3e} Eh/Bohr."
        )

    freq_line = _extract_field(r"VIBRATIONAL_FREQUENCIES_CM1:\s*([^\r\n]+)", "VIBRATIONAL_FREQUENCIES_CM1")
    freqs = [float(tok) for tok in freq_line.split()]
    imag_modes = [f for f in freqs if f < 0.0]
    if imag_modes:
        raise CCSDConvergenceError(
            f"PySCF geometry is not a true minimum: found {len(imag_modes)} imaginary modes {imag_modes}"
        )

    coord_idx = summary_text.rfind("FINAL_COORDINATES_ANGSTROM:")
    if coord_idx == -1:
        raise CCSDConvergenceError(f"Missing FINAL_COORDINATES_ANGSTROM in {p.name}")
    symbols: list[str] = []
    coords_list: list[list[float]] = []
    for cl in summary_text[coord_idx:].splitlines()[1:]:
        stripped = cl.strip()
        if not stripped or "TERMINATED NORMALLY" in stripped:
            break
        parts = stripped.split()
        if len(parts) >= 4 and re.match(r"^[A-Za-z]{1,2}$", parts[0]):
            symbols.append(parts[0])
            coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
        else:
            break
    if not symbols:
        raise CCSDConvergenceError(f"Could not parse coordinates from {p.name}")

    coords_arr = np.asarray(coords_list, dtype=np.float64)
    masses_u = np.array([get_exact_isotope_mass(s) for s in symbols], dtype=np.float64)
    moments_ie, rot_be = rotational_constants_from_geometry(coords_arr, masses_u)
    internals = compute_internal_geometry_metrics(symbols, coords_arr)

    xyz_file_str: str | None = None
    xyz_digest: str | None = None
    if xyz_path is not None:
        xp = Path(xyz_path)
        if xp.is_file():
            xyz_file_str = str(xp)
            xyz_digest = _sha256(xp)

    return CCSDOptimizationResult(
        backend="pyscf_ccsd",
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        symbols=symbols,
        optimized_coords_angstrom=coords_list,
        electronic_energy_hartree=energy,
        t1_diagnostic=t1_diag,
        max_gradient_eh_bohr=max_grad,
        rms_gradient_eh_bohr=rms_grad,
        vibrational_frequencies_cm1=freqs,
        n_imaginary_modes=len(imag_modes),
        converged_marker_found=conv_found,
        normal_termination_found=term_found,
        rotational_constants_be_mhz=(float(rot_be[0]), float(rot_be[1]), float(rot_be[2])),
        principal_moments_ie_u_a2=(float(moments_ie[0]), float(moments_ie[1]), float(moments_ie[2])),
        internal_geometry=internals,
        output_file=str(p),
        output_sha256=_sha256(p),
        xyz_file=xyz_file_str,
        xyz_sha256=xyz_digest,
        engine_version=f"PySCF {pyscf_ver} + geomeTRIC {geom_ver}",
    )


def run_pyscf_ccsd_optimization(
    work_dir: Path | str,
    job_name: str,
    symbols: list[str],
    initial_coords_ang: np.ndarray,
    *,
    basis: str = "cc-pVDZ",
    charge: int = 0,
    multiplicity: int = 1,
    wsl_distro: str = "Ubuntu",
    wsl_python: str = DEFAULT_WSL_PYSCF_PYTHON,
    timeout_seconds: int = 1800,
) -> CCSDOptimizationResult:
    """Execute a genuine CPU PySCF CCSD analytic-gradient + geomeTRIC optimization."""
    wd = Path(work_dir).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    script_path = wd / f"{job_name}_pyscf_runner.py"
    cfg_path = wd / f"{job_name}_pyscf_cfg.json"
    log_path = wd / f"{job_name}.log"
    xyz_path = wd / f"{job_name}.xyz"
    res_path = wd / f"{job_name}_result.json"

    masses_u = [get_exact_isotope_mass(s) for s in symbols]
    script_path.write_text(_pyscf_runner_script(), encoding="utf-8")

    # Decide native Python vs WSL2 bridge
    has_native_pyscf = False
    try:
        import geometric  # noqa: F401
        import pyscf  # noqa: F401

        has_native_pyscf = True
    except ImportError:
        has_native_pyscf = False

    if has_native_pyscf:
        cfg_payload: dict[str, Any] = {
            "symbols": symbols,
            "coords_ang": np.asarray(initial_coords_ang, dtype=np.float64).tolist(),
            "masses_u": masses_u,
            "basis": basis,
            "charge": charge,
            "multiplicity": multiplicity,
            "log_path": str(log_path),
            "xyz_path": str(xyz_path),
            "res_path": str(res_path),
        }
        cfg_path.write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")
        cmd = [sys.executable, str(script_path), str(cfg_path)]
    elif os.name == "nt":
        cfg_payload = {
            "symbols": symbols,
            "coords_ang": np.asarray(initial_coords_ang, dtype=np.float64).tolist(),
            "masses_u": masses_u,
            "basis": basis,
            "charge": charge,
            "multiplicity": multiplicity,
            "log_path": _win_to_wsl_path(log_path),
            "xyz_path": _win_to_wsl_path(xyz_path),
            "res_path": _win_to_wsl_path(res_path),
        }
        cfg_path.write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")
        cmd = [
            "wsl",
            "-d",
            wsl_distro,
            "--",
            wsl_python,
            _win_to_wsl_path(script_path),
            _win_to_wsl_path(cfg_path),
        ]
    else:
        raise RuntimeError(
            "PySCF + geomeTRIC are not installed in this Python environment and WSL2 is unavailable."
        )

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        creationflags=WIN_SUBPROCESS_FLAGS,
    )
    if proc.returncode != 0:
        raise CCSDConvergenceError(
            f"PySCF CCSD execution failed (rc={proc.returncode}): {proc.stderr[-1500:]}"
        )

    return parse_pyscf_ccsd_output(
        log_path,
        xyz_path=xyz_path if xyz_path.is_file() else None,
        charge=charge,
        multiplicity=multiplicity,
    )
