"""
kinstress.py - kinematic off-fault stress + Drucker-Prager screening (homogeneous full space).

Sources : vertical strike-slip subfaults (point moment tensors), raised-cosine slip history
          D(t) = Dfinal * 0.5*(1-cos(pi*(t-t0)/T)),  t0 = rupture time, T = rise time.
Medium  : infinite homogeneous (alpha, beta, rho)  -> no free surface, no layering.
Output  : stress-change time series at receivers; peak / final Drucker-Prager exceedance.

Sign convention: compression NEGATIVE (SeisSol).  Coordinates x east, y north, z UP (z<0 in rock).
"""
import math
import numpy as np
import numba as nb
from _kernel_gen import coef, coef_u


# ---------------------------------------------------------------- slip-history functions
@nb.njit(cache=True, inline='always')
def _P5(u, T, M0, out, off):
    """out[off+0..4] = M0 * (F2, F1, F, dF/dt, d2F/dt2)(u) for raised-cosine slip, i.e. P_{-2..2}."""
    if u <= 0.0:
        for k in range(5):
            out[off + k] = 0.0
    elif u < T:
        w = math.pi / T
        s = math.sin(w * u)
        c = math.cos(w * u)
        out[off + 0] = M0 * 0.5 * (0.5 * u * u + (c - 1.0) / (w * w))
        out[off + 1] = M0 * 0.5 * (u - s / w)
        out[off + 2] = M0 * 0.5 * (1.0 - c)
        out[off + 3] = M0 * 0.5 * w * s
        out[off + 4] = M0 * 0.5 * w * w * c
    else:
        w = math.pi / T
        v = u - T
        out[off + 0] = M0 * (T * T / 4.0 - 1.0 / (w * w) + 0.5 * T * v + 0.5 * v * v)
        out[off + 1] = M0 * (0.5 * T + v)
        out[off + 2] = M0
        out[off + 3] = 0.0
        out[off + 4] = 0.0


# ---------------------------------------------------------------- core accumulation
@nb.njit(cache=True)
def _add_source(rx, ry, rz, row, med, eps, dt, nt, acc, step, C, P):
    """Add one point source (row = x y z tx ty M0 t0 T) to acc[nt,6] / static-step array."""
    alpha, beta, rho, lam, mu = med[0], med[1], med[2], med[3], med[4]
    dx = rx - row[0]
    dy = ry - row[1]
    dz = rz - row[2]
    r0 = math.sqrt(dx * dx + dy * dy + dz * dz)
    # softening: push the receiver radially outward to r_eff (keeps direction, bounds the near field)
    r = math.sqrt(r0 * r0 + eps * eps)
    f = r / r0
    x = dx * f
    y = dy * f
    z = dz * f
    tx = row[3]
    ty = row[4]
    nx = -ty
    ny = tx
    M0 = row[5]
    t0 = row[6]
    T = row[7]
    coef(x, y, z, r, alpha, beta, rho, lam, mu, 2.0 * tx * nx, 2.0 * ty * ny, tx * ny + ty * nx, C)
    ta = r / alpha
    tb = r / beta
    k0 = int(math.ceil((t0 + ta) / dt))
    k1 = int(math.ceil((t0 + T + tb) / dt))       # from k1 on, the response is static
    if k0 < 0:
        k0 = 0
    kk = k1 if k1 < nt else nt
    for k in range(k0, kk):
        t = k * dt
        _P5(t - ta - t0, T, M0, P, 0)
        _P5(t - tb - t0, T, M0, P, 5)
        for c in range(6):
            v = 0.0
            for j in range(10):
                v += C[c, j] * P[j]
            acc[k, c] += v
    if k1 < nt:
        _P5(T + tb - ta, T, M0, P, 0)
        _P5(T, T, M0, P, 5)
        for c in range(6):
            v = 0.0
            for j in range(10):
                v += C[c, j] * P[j]
            step[k1, c] += v


@nb.njit(cache=True)
def _accumulate(rx, ry, rz, fine, coarse, med, dt, nt, r_near, rcut, eps, acc, step, C, P):
    """Stress-change history of all sources at one receiver -> acc[nt,6].
    Two-level source description: coarse[nb,8] (block point sources, used when the block center is
    farther than r_near) and fine[nb,nc,8] (children, used inside r_near).  Rows with M0==0 are skipped."""
    for k in range(nt + 1):
        for c in range(6):
            step[k, c] = 0.0
    for b in range(coarse.shape[0]):
        dx = rx - coarse[b, 0]
        dy = ry - coarse[b, 1]
        dz = rz - coarse[b, 2]
        db = math.sqrt(dx * dx + dy * dy + dz * dz)
        if db > rcut:
            continue
        if db > r_near:
            if coarse[b, 5] != 0.0:
                _add_source(rx, ry, rz, coarse[b], med, 0.0, dt, nt, acc, step, C, P)
        else:
            for c in range(fine.shape[1]):
                if fine[b, c, 5] != 0.0:
                    _add_source(rx, ry, rz, fine[b, c], med, eps, dt, nt, acc, step, C, P)
    run = np.zeros(6)
    for k in range(nt):
        for c in range(6):
            run[c] += step[k, c]
            acc[k, c] += run[c]


@nb.njit(cache=True, parallel=True)
def stress_history(rec, fine, coarse, med, dt, nt, r_near, rcut, eps):
    """Stress-change history (Pa) at receivers rec[n,3] -> out[n,nt,6] (xx yy zz xy yz xz)."""
    n = rec.shape[0]
    out = np.zeros((n, nt, 6))
    for i in nb.prange(n):
        step = np.zeros((nt + 1, 6))
        C = np.zeros((6, 10))
        P = np.zeros(10)
        acc = np.zeros((nt, 6))
        _accumulate(rec[i, 0], rec[i, 1], rec[i, 2], fine, coarse, med, dt, nt, r_near, rcut, eps, acc, step, C, P)
        out[i] = acc
    return out


def stress_history_flat(rec, src, med, dt, nt, rcut=1e12, eps=0.0):
    """Single-level convenience wrapper (validation): src[n,8]."""
    return stress_history(np.ascontiguousarray(rec, float), np.ascontiguousarray(src[:, None, :], float),
                          np.ascontiguousarray(src, float), med, dt, nt, 1e30, rcut, eps)


@nb.njit(cache=True)
def _dp(s, coh, sphi, cphi, pf):
    """Drucker-Prager exceedance F = sqrt(J2) - (coh*cos(phi) - (sigma_m + pf)*sin(phi)); compression<0.
    Returns F, sqrt(J2), sigma_m."""
    sm = (s[0] + s[1] + s[2]) / 3.0
    a = s[0] - sm
    b = s[1] - sm
    c = s[2] - sm
    j2 = 0.5 * (a * a + b * b + c * c) + s[3] * s[3] + s[4] * s[4] + s[5] * s[5]
    sj = math.sqrt(j2)
    return sj - (coh * cphi - (sm + pf) * sphi), sj, sm


@nb.njit(cache=True, parallel=True)
def yield_screen(rec, init, coh, phi, pf, fine, coarse, med, dt, nt, r_near, rcut, eps, kidx, nsus):
    """Peak-in-time and final Drucker-Prager exceedance at each receiver.
    init[n,6] initial stress (Pa, compression<0); coh[n] cohesion (Pa); phi[n] friction angle (rad);
    pf[n] pore pressure (Pa, compression-negative stress convention => pass as positive pressure).
    Returns peakF, tpeak, finalF, peakSJ2, dsm_min (most compressive mean-stress change), Fstatic0, F at the kidx steps,
    final stress change (n,6), the stress change at the kidx steps (n,nk,6), the running maximum of F up to each kidx step (n,nk),
    and the SUSTAINED peak F (n) and its running maximum at the kidx steps (n,nk): the largest value of min(F over nsus consecutive
    time steps), i.e. F > 0 there means F stayed above 0 for at least nsus steps (the SeisSol viscoplastic time Tv, nsus = ceil(Tv/dt))."""
    n = rec.shape[0]
    peakF = np.empty(n)
    tpeak = np.empty(n)
    finF = np.empty(n)
    peakSJ = np.empty(n)
    dsm_min = np.empty(n)
    F0 = np.empty(n)
    Fsl = np.zeros((n, kidx.shape[0]))
    Fmx = np.zeros((n, kidx.shape[0]))
    Fsus = np.empty(n); Fsus_sl = np.zeros((n, kidx.shape[0]))
    dsl = np.zeros((n, kidx.shape[0], 6))
    dfin = np.empty((n, 6))
    for i in nb.prange(n):
        Fk = np.empty(nt)
        step = np.zeros((nt + 1, 6))
        C = np.zeros((6, 10))
        P = np.zeros(10)
        acc = np.zeros((nt, 6))
        _accumulate(rec[i, 0], rec[i, 1], rec[i, 2], fine, coarse, med, dt, nt, r_near, rcut, eps, acc, step, C, P)
        sp_ = math.sin(phi[i])
        cp_ = math.cos(phi[i])
        s = np.empty(6)
        F0[i], _, _ = _dp(init[i], coh[i], sp_, cp_, pf[i])
        best = -1e30
        tb = 0.0
        bsj = 0.0
        dmin = 0.0
        for k in range(nt):
            for c in range(6):
                s[c] = init[i, c] + acc[k, c]
            F, sj, sm = _dp(s, coh[i], sp_, cp_, pf[i])
            Fk[k] = F
            if F > best:
                best = F
                tb = k * dt
                bsj = sj
            dsm = (acc[k, 0] + acc[k, 1] + acc[k, 2]) / 3.0
            if dsm < dmin:
                dmin = dsm
        for j in range(kidx.shape[0]):
            for c in range(6):
                s[c] = init[i, c] + acc[kidx[j], c]
                dsl[i, j, c] = acc[kidx[j], c]
            Fsl[i, j], _, _ = _dp(s, coh[i], sp_, cp_, pf[i])
            m_ = -1e30
            for k in range(kidx[j] + 1):
                if Fk[k] > m_:
                    m_ = Fk[k]
            Fmx[i, j] = m_
        ns_ = max(1, min(nsus, nt))
        Fsk = np.empty(nt)
        for k in range(nt):
            if k < ns_ - 1:
                Fsk[k] = -1e30
            else:
                mm = Fk[k]
                for q in range(1, ns_):
                    if Fk[k - q] < mm:
                        mm = Fk[k - q]
                Fsk[k] = mm
        mx = -1e30
        for k in range(nt):
            if Fsk[k] > mx:
                mx = Fsk[k]
        Fsus[i] = mx
        for j in range(kidx.shape[0]):
            m_ = -1e30
            for k in range(kidx[j] + 1):
                if Fsk[k] > m_:
                    m_ = Fsk[k]
            Fsus_sl[i, j] = m_
        for c in range(6):
            s[c] = init[i, c] + acc[nt - 1, c]
            dfin[i, c] = acc[nt - 1, c]
        Ff, _, _ = _dp(s, coh[i], sp_, cp_, pf[i])
        peakF[i] = best
        tpeak[i] = tb
        finF[i] = Ff
        peakSJ[i] = bsj
        dsm_min[i] = dmin
    return peakF, tpeak, finF, peakSJ, dsm_min, F0, Fsl, dfin, dsl, Fmx, Fsus, Fsus_sl


@nb.njit(cache=True)
def displacement_history(rx, ry, rz, src, med, dt, nt):
    """Displacement time series at one point (full evaluation, used for validation only)."""
    alpha, beta, rho, lam, mu = med[0], med[1], med[2], med[3], med[4]
    out = np.zeros((nt, 3))
    C = np.zeros((3, 10))
    P = np.zeros(10)
    for s in range(src.shape[0]):
        x = rx - src[s, 0]
        y = ry - src[s, 1]
        z = rz - src[s, 2]
        r = math.sqrt(x * x + y * y + z * z)
        tx = src[s, 3]
        ty = src[s, 4]
        nx = -ty
        ny = tx
        coef_u(x, y, z, r, alpha, beta, rho, lam, mu, 2 * tx * nx, 2 * ty * ny, tx * ny + ty * nx, C)
        for k in range(nt):
            t = k * dt
            _P5(t - r / alpha - src[s, 6], src[s, 7], src[s, 5], P, 0)
            _P5(t - r / beta - src[s, 6], src[s, 7], src[s, 5], P, 5)
            for c in range(3):
                v = 0.0
                for j in range(10):
                    v += C[c, j] * P[j]
                out[k, c] += v
    return out


def medium(alpha=6000.0, beta=3464.0, rho=2700.0):
    mu = rho * beta ** 2
    lam = rho * (alpha ** 2 - 2 * beta ** 2)
    return np.array([alpha, beta, rho, lam, mu])
