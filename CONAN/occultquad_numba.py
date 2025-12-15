"""Numba-accelerated Mandel & Agol (2002) occultquad implementation.

This module provides a drop-in style function `occultquad_numba` that
computes the same quadratic limb-darkened transit/occultation light curve
as `OccultQuadPy.occultquad`, but with the main loop JIT-compiled using
Numba for a substantial speed-up.

The formulas follow the Python translation in `occultquad_pya.py`, which
itself is based on the original Fortran routines by Mandel & Agol (2002).
"""

from __future__ import annotations

import numpy as np
from numpy import sqrt, log

try:  # Numba is an optional dependency
    import numba as nb
except ImportError as exc:  # pragma: no cover - import error is user-facing
    raise ImportError(
        "Numba is required for CONAN.CONAN.occultquad_numba. "
        "Install it with `pip install numba` or `conda install numba`."
    ) from exc


@nb.njit(cache=True)
def _rf_numba(x, y, z):
    """Legendre form incomplete elliptic integral RF (Numba version).

    Translation of `OccultQuadPy.rf` with domain checks omitted under the
    assumption that calling code passes valid physical arguments.
    """
    ERRTOL = 0.08
    TINY = 1.5e-38
    BIG = 3.0e37

    # The following domain checks are kept but return 0.0 instead of raising
    # to keep the function nopython-friendly. For valid transit inputs these
    # branches should never be taken.
    if (min(x, y, z) < 0.0) or (min(x + y, x + z, y + z) < TINY) or (max(x, y, z) > BIG):
        return 0.0

    xt, yt, zt = x, y, z

    while True:
        sqrtx = sqrt(xt)
        sqrty = sqrt(yt)
        sqrtz = sqrt(zt)
        alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        zt = 0.25 * (zt + alamb)
        ave = (xt + yt + zt) / 3.0
        delx = (ave - xt) / ave
        dely = (ave - yt) / ave
        delz = (ave - zt) / ave
        if max(abs(delx), abs(dely), abs(delz)) <= ERRTOL:
            break

    C1 = 1.0 / 24.0
    C2 = 0.1
    C3 = 3.0 / 44.0
    C4 = 1.0 / 14.0

    e2 = delx * dely - delz * delz
    e3 = delx * dely * delz
    rf = (1.0 + (C1 * e2 - C2 - C3 * e3) * e2 + C4 * e3) / sqrt(ave)
    return rf


@nb.njit(cache=True)
def _rc_numba(x, y):
    """RC elliptic integral (Numba version of `OccultQuadPy.rc`)."""
    SQRTNY = 1.3e-19
    ERRTOL = 0.04
    TINY = 1.69e-38
    BIG = 3.0e37

    TNBG = TINY * BIG
    COMP1 = 2.236 / SQRTNY
    COMP2 = TNBG * TNBG / 25.0

    if (x < 0.0) or (y == 0.0) or (x + abs(y) < TINY) or (x + abs(y) > BIG):
        return 0.0
    if (y < -COMP1) and (x > 0.0) and (x < COMP2):
        return 0.0

    if y > 0.0:
        xt = x
        yt = y
        w = 1.0
    else:
        xt = x - y
        yt = -y
        w = sqrt(x) / sqrt(xt)

    while True:
        alamb = 2.0 * sqrt(xt) * sqrt(yt) + yt
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        ave = (xt + yt + yt) / 3.0
        s = (yt - ave) / ave
        if abs(s) <= ERRTOL:
            break

    C1 = 0.3
    C2 = 1.0 / 7.0
    C3 = 0.375
    C4 = 9.0 / 22.0

    rc = w * (1.0 + s * s * (C1 + s * (C2 + s * (C3 + s * C4)))) / sqrt(ave)
    return rc


@nb.njit(cache=True)
def _rj_numba(x, y, z, p):
    """RJ elliptic integral (Numba version of `OccultQuadPy.rj`)."""
    ERRTOL = 0.05
    TINY = 2.5e-13
    BIG = 9.0e11

    if (min(x, y, z) < 0.0) or (min(x + y, x + z, y + z, abs(p)) < TINY):
        return 0.0
    if max(x, y, z, abs(p)) > BIG:
        return 0.0

    summe = 0.0
    fac = 1.0

    if p > 0.0:
        xt = x
        yt = y
        zt = z
        pt = p
    else:
        xt = min(x, y, z)
        zt = max(x, y, z)
        yt = x + y + z - xt - zt
        a = 1.0 / (yt - p)
        b = a * (zt - yt) * (yt - xt)
        pt = yt + b
        rho = xt * zt / yt
        tau = p * pt / yt
        rcx = _rc_numba(rho, tau)

    while True:
        sqrtx = sqrt(xt)
        sqrty = sqrt(yt)
        sqrtz = sqrt(zt)
        alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz
        alpha = (pt * (sqrtx + sqrty + sqrtz) + sqrtx * sqrty * sqrtz) ** 2
        beta = pt * (pt + alamb) ** 2
        summe += fac * _rc_numba(alpha, beta)
        fac *= 0.25
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        zt = 0.25 * (zt + alamb)
        pt = 0.25 * (pt + alamb)
        ave = 0.2 * (xt + yt + zt + pt + pt)
        delx = (ave - xt) / ave
        dely = (ave - yt) / ave
        delz = (ave - zt) / ave
        delp = (ave - pt) / ave
        if max(abs(delx), abs(dely), abs(delz), abs(delp)) <= ERRTOL:
            break

    ea = delx * (dely + delz) + dely * delz
    eb = delx * dely * delz
    ec = delp * delp
    ed = ea - 3.0 * ec
    ee = eb + 2.0 * delp * (ea - ec)

    C1 = 3.0 / 14.0
    C2 = 1.0 / 3.0
    C3 = 3.0 / 22.0
    C4 = 3.0 / 26.0
    C5 = 0.75 * C3
    C6 = 1.5 * C4
    C7 = 0.5 * C2
    C8 = C3 + C3

    rj = 3.0 * summe + fac * (
        1.0
        + ed * (-C1 + C5 * ed - C6 * ee)
        + eb * (C7 + delp * (-C8 + delp * C4))
        + delp * ea * (C2 - delp * C3)
        - C2 * delp * ec
    ) / (ave * sqrt(ave))

    if p < 0.0:
        rj = a * (b * rj + 3.0 * (rcx - _rf_numba(xt, yt, zt)))

    return rj


@nb.njit(cache=True)
def _ellk_numba(k):
    """Complete elliptic integral of 1st kind (Hastings approx)."""
    m1 = 1.0 - k * k
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012
    ek1 = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ek2 = (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * log(m1)
    return ek1 - ek2


@nb.njit(cache=True)
def _ellec_numba(k):
    """Complete elliptic integral of 2nd kind (Hastings approx)."""
    m1 = 1.0 - k * k
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639
    ee1 = 1.0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ee2 = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * log(1.0 / m1)
    return ee1 + ee2


@nb.njit(cache=True)
def _occultquad_kernel(z0, u1, u2, p, nz=None):
    """Numba kernel implementing OccultQuadPy.occultquad logic.

    Parameters
    ----------
    z0 : 1D float64 array
    u1, u2, p : scalars

    Returns
    -------
    muo1, mu0 : 1D float64 arrays
    """
    nz = z0.size

    if abs(p - 0.5) < 1e-3:
        p = 0.5

    omega = 1.0 - u1 / 3.0 - u2 / 6.0
    pi = np.pi

    lambdad = np.zeros(nz, dtype=np.float64)
    etad = np.zeros(nz, dtype=np.float64)
    lambdae = np.zeros(nz, dtype=np.float64)

    for i in range(nz):
        z = z0[i]
        x1 = (p - z) * (p - z)
        x2 = (p + z) * (p + z)
        x3 = p * p - z * z

        # Case I: source unocculted
        if z >= 1.0 + p:
            lambdad[i] = 0.0
            etad[i] = 0.0
            lambdae[i] = 0.0
            continue

        # Case II: source completely occulted
        if (p >= 1.0) and (z <= p - 1.0):
            lambdad[i] = 1.0
            etad[i] = 1.0
            lambdae[i] = 1.0
            continue

        # Partly occulted, crosses limb (Eq. 26)
        if (z >= abs(1.0 - p)) and (z <= 1.0 + p):
            kap1 = np.arccos(min((1.0 - p * p + z * z) / (2.0 * z), 1.0))
            kap0 = np.arccos(min((p * p + z * z - 1.0) / (2.0 * p * z), 1.0))
            lambdae[i] = p * p * kap0 + kap1
            tmp = 4.0 * z * z - (1.0 + z * z - p * p) ** 2
            if tmp < 0.0:
                tmp = 0.0
            lambdae[i] = (lambdae[i] - 0.5 * sqrt(tmp)) / pi

        # Transiting, not fully covering
        if z <= 1.0 - p:
            lambdae[i] = p * p

        # Edge at origin - special expressions
        if abs(z - p) < 1e-4 * (z + p):
            if z >= 0.5:
                q = 0.5 / p
                Kk = _ellk_numba(q)
                Ek = _ellec_numba(q)
                lambdad[i] = 1.0 / 3.0 + 16.0 * p / 9.0 / pi * (2.0 * p * p - 1.0) * Ek - (
                    (32.0 * p ** 4 - 20.0 * p * p + 3.0) / 9.0 / pi / p
                ) * Kk
                # eta_1
                etad[i] = 1.0 / 2.0 / pi * (
                    kap1
                    + p * p * (p * p + 2.0 * z * z) * kap0
                    - (1.0 + 5.0 * p * p + z * z) / 4.0 * sqrt((1.0 - x1) * (x2 - 1.0))
                )
                if p == 0.5:
                    lambdad[i] = 1.0 / 3.0 - 4.0 / pi / 9.0
                    etad[i] = 3.0 / 32.0
                continue
            else:
                q = 2.0 * p
                Kk = _ellk_numba(q)
                Ek = _ellec_numba(q)
                lambdad[i] = 1.0 / 3.0 + 2.0 / 9.0 / pi * (
                    4.0 * (2.0 * p * p - 1.0) * Ek + (1.0 - 4.0 * p * p) * Kk
                )
                etad[i] = p * p / 2.0 * (p * p + 2.0 * z * z)
                continue

        # Case III: partly occults and crosses limb
        cond1 = (z > 0.5 + abs(p - 0.5)) and (z < 1.0 + p)
        cond2 = (p > 0.5) and (z > abs(1.0 - p) * 1.0001) and (z < p)
        if cond1 or cond2:
            q = sqrt((1.0 - (p - z) * (p - z)) / (4.0 * z * p))
            Kk = _ellk_numba(q)
            Ek = _ellec_numba(q)
            n = 1.0 / x1 - 1.0
            Pk = Kk - n / 3.0 * _rj_numba(0.0, 1.0 - q * q, 1.0, 1.0 + n)
            lambdad[i] = 1.0 / 9.0 / pi / sqrt(p * z) * (
                ((1.0 - x2) * (2.0 * x2 + x1 - 3.0) - 3.0 * x3 * (x2 - 2.0)) * Kk
                + 4.0 * p * z * (z * z + 7.0 * p * p - 4.0) * Ek
                - 3.0 * x3 / x1 * Pk
            )
            if z < p:
                lambdad[i] += 2.0 / 3.0
            etad[i] = 1.0 / 2.0 / pi * (
                kap1
                + p * p * (p * p + 2.0 * z * z) * kap0
                - (1.0 + 5.0 * p * p + z * z) / 4.0 * sqrt((1.0 - x1) * (x2 - 1.0))
            )
            continue

        # Case IV: transits the source
        if (p <= 1.0) and (z <= (1.0 - p) * 1.0001):
            q = sqrt((x2 - x1) / (1.0 - x1))
            Kk = _ellk_numba(q)
            Ek = _ellec_numba(q)
            n = x2 / x1 - 1.0
            Pk = Kk - n / 3.0 * _rj_numba(0.0, 1.0 - q * q, 1.0, 1.0 + n)
            lambdad[i] = 2.0 / 9.0 / pi / sqrt(1.0 - x1) * (
                (1.0 - 5.0 * z * z + p * p + x3 * x3) * Kk
                + (1.0 - x1) * (z * z + 7.0 * p * p - 4.0) * Ek
                - 3.0 * x3 / x1 * Pk
            )
            if z < p:
                lambdad[i] += 2.0 / 3.0
            if abs(p + z - 1.0) <= 1e-4:
                lambdad[i] = 2.0 / 3.0 / pi * np.arccos(1.0 - 2.0 * p) - 4.0 / 9.0 / pi * sqrt(
                    p * (1.0 - p)
                ) * (3.0 + 2.0 * p - 8.0 * p * p)
            etad[i] = p * p / 2.0 * (p * p + 2.0 * z * z)

    muo1 = 1.0 - ((1.0 - u1 - 2.0 * u2) * lambdae + (u1 + 2.0 * u2) * lambdad + u2 * etad) / omega
    mu0 = 1.0 - lambdae
    return muo1, mu0


def occultquad_numba(z0, u1, u2, p, nz=None):
    """Public wrapper for the Numba-accelerated occultquad.

    Parameters
    ----------
    z0 : array_like
        Impact parameters in units of stellar radius.
    u1, u2 : float
        Quadratic limb-darkening coefficients.
    p : float
        Planet/star radius ratio.
    nz : int, optional
        Number of values in z0. Determined here from z0, parameter
        introduced to maintain API.

    Returns
    -------
    muo1 : array
        Fraction of flux at each z0 (impact parameter) for a limb-darkened source
    mu0 : array
        Fraction of flux at each z0 (impact parameter) for a uniform source
    """
    
    z0_arr = np.asarray(z0, dtype=np.float64)
    u1_f = float(u1)
    u2_f = float(u2)
    p_f = float(p)
    return _occultquad_kernel(z0_arr, u1_f, u2_f, p_f)
