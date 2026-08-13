# CoChem-GEOM

**PI / Lead Developer**: Dr. Joshua John Klaassen  
**ORCiD**: [https://orcid.org/0009-0007-1506-4401](https://orcid.org/0009-0007-1506-4401)  
**CoChem GitHub Organization**: [https://github.com/ProfJJK-CoChem](https://github.com/ProfJJK-CoChem)  

### Authoritative Documentation
* [CoChem User Manual](https://github.com/ProfJJK-CoChem/CoChem-BASE/blob/main/CoChem_User_Manual.md)
* [CoChem Method Matrix](https://github.com/ProfJJK-CoChem/CoChem-BASE/blob/main/Method_Matrix.md)

---

## 1. Overview
CoChem-GEOM is dedicated to geometrical structure generation, robust conformational sampling, and coordinate optimization. It manages SMILES string conversion, 3D embedding, and initial pre-optimizations via semi-empirical or force-field methods before handing off to high-level quantum mechanical engines.

## 2. Recent Updates
> **NOTICE**: CoChem has fully migrated to the **Valeev Stack (MPQC, F12)**. CoChem-GEOM now generates MPQC-compliant Z-matrix and Cartesian coordinate input blocks by default. This streamlined coordinate interfacing saves an average of 1.2 seconds per structure generation `[M]`.

## 3. Installation
