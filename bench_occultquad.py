"""Benchmark Python vs Numba vs Fortran occultquad implementations.

Run this script from the project root (with CONAN importable) to compare
runtime and numerical agreement between:

- `CONAN.occultquad_pya.OccultQuadPy.occultquad` (pure Python)
- `CONAN.occultquad_numba.occultquad_numba` (Numba-accelerated)
- `CONAN.occultquad.occultquad` (Fortran/f2py extension, if available)
"""

from __future__ import annotations

import time

import numpy as np

from CONAN.occultquad_pya import OccultQuadPy
from CONAN.occultquad_numba import occultquad_numba

try:  # Fortran extension is optional
    from CONAN.occultquad import occultquad as occultquad_fortran
    HAS_FORTRAN = True
except Exception:  # pragma: no cover - environment without compiled extension
    occultquad_fortran = None
    HAS_FORTRAN = False


def run_benchmark(nz: int = 20000) -> None:
    z0 = np.linspace(0.0, 2.0, nz)
    u1, u2, p = 0.3, 0.2, 0.1

    print(f"Benchmarking with nz={nz} impact parameters")
    print(f"u1={u1}, u2={u2}, p={p}\n")

    # Pure Python implementation
    oq = OccultQuadPy()
    t0 = time.perf_counter()
    muo1_py, mu0_py = oq.occultquad(z0, u1, u2, p)
    t1 = time.perf_counter()
    t_py = t1 - t0

    # Fortran implementation (if available)
    t_fortran = None
    if HAS_FORTRAN:
        t_f0 = time.perf_counter()
        # Fortran wrapper signature matches usage in CONAN.models:
        # occultquad(z, u1, u2, p, nz) -> (muo1, mu0)
        muo1_f, mu0_f = occultquad_fortran(z0, u1, u2, p, nz)
        t_f1 = time.perf_counter()
        t_fortran = t_f1 - t_f0

    # Numba implementation: first call includes JIT compile time
    t2 = time.perf_counter()
    muo1_nb, mu0_nb = occultquad_numba(z0, u1, u2, p)
    t3 = time.perf_counter()
    t_nb_first = t3 - t2

    # Second call shows steady-state performance
    t4 = time.perf_counter()
    muo1_nb2, mu0_nb2 = occultquad_numba(z0, u1, u2, p)
    t5 = time.perf_counter()
    t_nb = t5 - t4

    # Consistency checks (Python vs Numba)
    max_diff_muo1_py_nb = float(np.max(np.abs(muo1_py - muo1_nb)))
    max_diff_mu0_py_nb = float(np.max(np.abs(mu0_py - mu0_nb)))

    print("Results (times in seconds):")
    print(f"  Python:            {t_py:8.4f}")
    if HAS_FORTRAN and t_fortran is not None:
        print(f"  Fortran:           {t_fortran:8.4f}")
    print(f"  Numba (1st call):  {t_nb_first:8.4f}  (includes JIT compilation)")
    print(f"  Numba (2nd call):  {t_nb:8.4f}")
    if t_nb > 0:
        print(f"  Speedup vs Python: {t_py / t_nb:8.2f}x (Numba steady state)")
        if HAS_FORTRAN and t_fortran is not None:
            print(f"  Speedup vs Fortran:{t_fortran / t_nb:8.2f}x (Numba vs Fortran)")

    print()
    print("Max absolute differences (Python vs Numba):")
    print(f"  max |muo1_py - muo1_nb| = {max_diff_muo1_py_nb:.3e}")
    print(f"  max |mu0_py  - mu0_nb|  = {max_diff_mu0_py_nb:.3e}")

    if HAS_FORTRAN and t_fortran is not None:
        max_diff_muo1_py_f = float(np.max(np.abs(muo1_py - muo1_f)))
        max_diff_mu0_py_f = float(np.max(np.abs(mu0_py - mu0_f)))
        max_diff_muo1_nb_f = float(np.max(np.abs(muo1_nb - muo1_f)))
        max_diff_mu0_nb_f = float(np.max(np.abs(mu0_nb - mu0_f)))

        print()
        print("Max absolute differences involving Fortran:")
        print(f"  max |muo1_py - muo1_f|  = {max_diff_muo1_py_f:.3e}")
        print(f"  max |mu0_py  - mu0_f|   = {max_diff_mu0_py_f:.3e}")
        print(f"  max |muo1_nb - muo1_f|  = {max_diff_muo1_nb_f:.3e}")
        print(f"  max |mu0_nb  - mu0_f|   = {max_diff_mu0_nb_f:.3e}")


if __name__ == "__main__":  # pragma: no cover
    run_benchmark()
