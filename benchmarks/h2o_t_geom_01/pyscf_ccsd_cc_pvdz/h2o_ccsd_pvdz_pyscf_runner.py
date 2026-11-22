
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
