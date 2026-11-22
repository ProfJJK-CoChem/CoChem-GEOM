"""Canonical CODATA physical constants and exact IUPAC/AME2020 nuclide masses.

Single source of truth for CoChem-GEOM rotational conversion factors and atomic
masses. Never duplicate or hardcode approximate conversion constants elsewhere.
"""

from __future__ import annotations

import re

# CODATA 2018/2022 rotational conversion constant:
# h / (8 * pi^2 * u * (1e-10 m)^2) * 1e-6  [MHz * u * Angstrom^2]
ROTATIONAL_CONSTANT_FACTOR_MHZ_U_A2: float = 505379.008435

# Fundamental mass constants (CODATA 2018/2022, in unified atomic mass units u)
ELECTRON_MASS_U: float = 5.48579909065e-4
PROTON_MASS_U: float = 1.007276466621
ELECTRON_TO_PROTON_MASS_RATIO: float = ELECTRON_MASS_U / PROTON_MASS_U

# Length and energy unit conversions
BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_KCAL_MOL: float = 627.5094740631

# Costain empirical systematic uncertainty constant (Costain 1966; van Eijck 1982)
# sigma_Costain(|r_s|) = COSTAIN_CONSTANT_A2 / |r_s|  [Angstrom]
COSTAIN_CONSTANT_A2: float = 0.0015

# Exact monoisotopic nuclear/atomic masses (AME2020 / IUPAC, in unified atomic mass units u)
EXACT_ISOTOPE_MASSES_U: dict[str, float] = {
    "H": 1.00782503223,
    "1H": 1.00782503223,
    "D": 2.01410177812,
    "2H": 2.01410177812,
    "T": 3.01604928132,
    "3H": 3.01604928132,
    "He": 4.00260325413,
    "4He": 4.00260325413,
    "Li": 7.0160034366,
    "7Li": 7.0160034366,
    "C": 12.0000000,
    "12C": 12.0000000,
    "13C": 13.00335483507,
    "14C": 14.003241988,
    "N": 14.00307400443,
    "14N": 14.00307400443,
    "15N": 15.00010889888,
    "O": 15.99491461957,
    "16O": 15.99491461957,
    "17O": 16.99913175650,
    "18O": 17.99915961286,
    "F": 18.99840316273,
    "19F": 18.99840316273,
    "Na": 22.9897692820,
    "23Na": 22.9897692820,
    "Si": 27.97692653465,
    "28Si": 27.97692653465,
    "P": 30.97376199842,
    "31P": 30.97376199842,
    "S": 31.9720711744,
    "32S": 31.9720711744,
    "33S": 32.9714589098,
    "34S": 33.967867004,
    "36S": 35.96708071,
    "Cl": 34.968852682,
    "35Cl": 34.968852682,
    "37Cl": 36.965902602,
    "Br": 78.9183376,
    "79Br": 78.9183376,
    "81Br": 80.9162897,
    "I": 126.9044719,
    "127I": 126.9044719,
}

# Natural isotopic abundances (percent, IUPAC)
ISOTOPE_NATURAL_ABUNDANCE_PCT: dict[str, float] = {
    "13C": 1.07,
    "14C": 1.0e-10,
    "2H": 0.0115,
    "D": 0.0115,
    "15N": 0.368,
    "17O": 0.038,
    "18O": 0.205,
    "33S": 0.75,
    "34S": 4.21,
    "36S": 0.02,
    "37Cl": 24.23,
    "81Br": 49.31,
}


def get_exact_isotope_mass(symbol: str, mass_number: int | None = None) -> float:
    """Return exact monoisotopic atomic mass in Daltons (u).

    Accepts canonical symbols ('O', 'H', 'D'), prefixed isotope labels ('18O', '13C',
    '2H'), or explicit (symbol, mass_number) pairs. Never falls back to average
    atomic weights.
    """
    raw = symbol.strip()
    if mass_number is None:
        m = re.match(r"^(\d+)([A-Za-z]+)$", raw)
        if m:
            mass_number = int(m.group(1))
            raw = m.group(2)
    key = f"{mass_number}{raw}" if mass_number is not None else raw
    if key in EXACT_ISOTOPE_MASSES_U:
        return EXACT_ISOTOPE_MASSES_U[key]
    if mass_number is None and raw in EXACT_ISOTOPE_MASSES_U:
        return EXACT_ISOTOPE_MASSES_U[raw]
    raise ValueError(
        f"No exact monoisotopic mass registered for symbol={symbol!r}, mass_number={mass_number!r}"
    )
