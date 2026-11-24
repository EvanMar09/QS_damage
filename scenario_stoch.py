"""
scenario_stoch.py - stochastic synthetic sources for the kinematic yielding screen.

  * smooth log-normal slip field with separate correlation lengths along strike / down dip
  * shallow slip deficit: slip at the surface is `ssd_pct` % of the mean slip below `ssd_depth`, rising back to 100 % at
    `ssd_depth` either abruptly or smoothly (`ssd_smooth` 0..1)
  * spatially variable rupture speed: v is drawn from a (truncated) normal pdf at nodes every
    `vr_seg` meters along strike (optionally also down dip) and the rupture time is the first-arrival
    (eikonal) time from the hypocenter through that speed field.
"""
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def taper_weights(S, Z, L, W, taper_s, taper_z):
    """Slip weights: sin^2 ramp over taper_s metres at both ends along strike, cos^2 ramp over taper_z metres above the fault base.
    A length <= 0 switches that taper off (slip stays at its full value up to the edge)."""
    ws = np.ones_like(S) if taper_s <= 0 else np.sin(0.5 * np.pi * np.clip(np.minimum(S, L - S) / taper_s, 0, 1)) ** 2
    wz = np.ones_like(Z) if taper_z <= 0 else np.cos(0.5 * np.pi * np.clip((Z - (W - taper_z)) / taper_z, 0, 1)) ** 2
    return ws, wz


def correlated_noise(shape, ds, dz, corr_s, corr_z, rng):
    """zero-mean, unit-variance Gaussian random field with Gaussian-filter correlation lengths (m)."""
    n = rng.standard_normal(shape)
    n = gaussian_filter(n, (max(corr_s, 1e-9) / ds, max(corr_z, 1e-9) / dz), mode='reflect')
    return (n - n.mean()) / n.std()



def gp_rise_time(D, z, M0, mean=None, eta=0.8, cv=0.85, depth_factor=True, shallow=2.0, corr_s=2000.0, corr_z=2000.0, ds=100.0, dz=100.0,
                 seed=0, min_frac=0.25, area_frac=0.1):
    """Rise time T(s,z) after Graves & Pitarka (2016, BSSA 106, Appendix; GP15.4).
        tau_i = b(z_i) * k * sqrt(n_i)
    n_i  : positive field with mean 1 and coefficient of variation `cv` (0.85), partly correlated (coefficient `eta`) with the slip:
           ln n = eta*ln(D/Dmean) + sqrt(1-eta^2)*sigma*g - (1-eta^2) sigma^2/2,  g = smooth Gaussian noise (correlation lengths corr_s, corr_z),
           sigma^2 = ln(1+cv^2).   eta = 1 gives tau ~ sqrt(local slip) exactly (GP2010); eta = 0 is independent of slip.  (An approximation of the
           paper's correlation-in-Gaussian-space; the same in the two limits.)
    b(z) : 2 above 5 km, 1 between 8 and 15 km, 2 below 18 km, linear in between (velocity-strengthening shallow/deep zones); depth_factor=False -> 1.
    k    : fixes the fault-average rise time (cells with D >= area_frac*max) to  tau_A = alpha_T * 1.6e-9 * M0^(1/3)  (M0 in dyne-cm, Somerville et al. 1999
           as modified by GP; alpha_T = 1 for a vertical strike-slip fault),  or to `mean` seconds if given.
    A floor of min_frac*tau_A keeps the rise time of nearly unslipped cells from going to zero.
    Returns T (same shape as D) and a dict of the numbers used."""
    D = np.asarray(D, float); ns_, nz_ = D.shape
    Dmax = D.max(); act = D >= area_frac * Dmax
    tauA = float(mean) if mean else 1.6e-9 * (M0 * 1e7) ** (1.0 / 3.0)
    sig = np.sqrt(np.log(1.0 + cv ** 2))
    Dm = D[act].mean()
    lnD = np.log(np.maximum(D, 0.01 * Dmax) / Dm)
    g = correlated_noise(D.shape, ds, dz, corr_s, corr_z, np.random.default_rng(seed)) if eta < 1.0 else np.zeros(D.shape)
    e = float(np.clip(eta, 0.0, 1.0))
    ln_n = e * lnD + np.sqrt(1.0 - e ** 2) * sig * g - 0.5 * (1.0 - e ** 2) * sig ** 2
    n = np.exp(ln_n)
    zz = np.asarray(z, float)
    b = np.ones_like(zz)
    if depth_factor:
        b = np.where(zz <= 5000.0, shallow, np.where(zz < 8000.0, shallow + (1.0 - shallow) * (zz - 5000.0) / 3000.0, 1.0))
        b = np.where(zz >= 15000.0, 1.0 + (shallow - 1.0) * np.minimum((zz - 15000.0) / 3000.0, 1.0), b)
    raw = b[None, :] * np.sqrt(n)
    T = raw * tauA / raw[act].mean()
    for _ in range(3):                                    # floor, then re-pin the mean
        T = np.maximum(T, min_frac * tauA)
        T *= tauA / T[act].mean()
    T = np.maximum(T, min_frac * tauA)
    info = dict(tau_A=tauA, mean=float(T[act].mean()), tmin=float(T[act].min()), tmax=float(T[act].max()), p10=float(np.percentile(T[act], 10)), p90=float(np.percentile(T[act], 90)),
                corr_slip=float(np.corrcoef(np.sqrt(D[act]), T[act])[0, 1]))
    return T, info


def depth_ramp(z, depth, ratio_surface, smooth=0.0):
    """Depth multiplier g(z): equals `ratio_surface` at the surface and 1 at z >= `depth`.
    smooth = 0 : abrupt step at `depth` (g = ratio_surface above, 1 below).
    smooth = 1 : cosine transition spread over the whole 0..depth layer (g(0) = ratio_surface, g(depth) = 1).
    0 < smooth < 1 : flat at ratio_surface down to depth*(1-smooth), then a cosine transition ending at `depth`."""
    z = np.asarray(z, float)
    smooth = float(np.clip(smooth, 0.0, 1.0))
    if smooth <= 0.0:
        return np.where(z < depth, ratio_surface, 1.0)
    z1 = depth * (1.0 - smooth)
    r = np.clip((z - z1) / (depth - z1), 0.0, 1.0)
    return ratio_surface + (1.0 - ratio_surface) * 0.5 * (1.0 - np.cos(np.pi * r))


def apply_shallow_deficit(D, z, ssd_pct, ssd_depth, smooth=0.0):
    """Shallow slip deficit.  Slip at the surface is ~ssd_pct % of the column's mean slip below `ssd_depth`, and rises back
    to the unmodified slip at z = ssd_depth.  `smooth` (0..1) sets how that rise is spread through the layer: 0 = abrupt step
    at ssd_depth (the layer mean is then exactly ssd_pct % of the mean below), 1 = gradual over the whole layer
    (no sharp slip gradient).  The surface value is exact when the unmodified slip is uniform within the layer, otherwise approximate.
    ssd_pct = 100 -> no change."""
    top = z < ssd_depth
    if not top.any() or ssd_pct >= 100.0:
        return D
    D = D.copy()
    below = z >= ssd_depth
    mb = D[:, below].mean(axis=1)
    ma = D[:, top].mean(axis=1)
    c = np.where(ma > 1e-9, mb / np.maximum(ma, 1e-9), 1.0)           # column match: layer level = mean below
    ratio = ssd_pct / 100.0
    g = depth_ramp(z, ssd_depth, ratio, smooth)                        # ratio at the surface -> 1 at ssd_depth
    w = np.where(ratio < 1.0, (g - ratio) / (1.0 - ratio), 1.0)        # 0 (surface level) .. 1 (full depth level)
    f0 = c * ratio                                                     # factor applied at surface level
    F = f0[:, None] + (1.0 - f0[:, None]) * w[None, :]
    return D * np.where(top[None, :], F, 1.0)


def speed_field(s, z, L, W, vr, vr_std, seg, seg_z, vmin, vmax, rng, step=False):
    """rupture-speed grid v[ns,nz].  Nodes every `seg` m along strike (and `seg_z` m down dip, 0 = whole width),
    v ~ N(vr, vr_std) clipped to [vmin, vmax]; linear interpolation between nodes (or piecewise constant if step)."""
    if vr_std <= 0:
        return np.full((len(s), len(z)), float(vr))
    ns_n = max(int(np.ceil(L / seg)), 1) + 1
    nodes_s = np.linspace(0, L, ns_n)
    if seg_z and seg_z > 0:
        nodes_z = np.linspace(0, W, max(int(np.ceil(W / seg_z)), 1) + 1)
    else:
        nodes_z = np.array([0.0, W])
    V = np.clip(rng.normal(vr, vr_std, (len(nodes_s), len(nodes_z))), vmin, vmax)
    if not (seg_z and seg_z > 0):
        V[:, 1] = V[:, 0]
    S, Z = np.meshgrid(s, z, indexing='ij')
    if step:
        i = np.clip(np.floor(S / (L / (ns_n - 1))).astype(int), 0, ns_n - 2)
        j = np.clip(np.searchsorted(nodes_z, Z, side='right') - 1, 0, len(nodes_z) - 2)
        return V[i, j]
    f = RegularGridInterpolator((nodes_s, nodes_z), V, bounds_error=False, fill_value=None)
    return f(np.stack([S.ravel(), Z.ravel()], -1)).reshape(S.shape)


@njit(cache=True)
def _sweep(t, f, hs, hz, n_iter):
    ns, nz = t.shape
    A = 1.0 / hs**2 + 1.0 / hz**2
    for it in range(n_iter):
        change = 0.0
        for sweep in range(4):
            i0, i1, di = (0, ns, 1) if sweep in (0, 1) else (ns - 1, -1, -1)
            j0, j1, dj = (0, nz, 1) if sweep in (0, 2) else (nz - 1, -1, -1)
            i = i0
            while i != i1:
                j = j0
                while j != j1:
                    a = min(t[i - 1, j] if i > 0 else 1e30, t[i + 1, j] if i < ns - 1 else 1e30)
                    b = min(t[i, j - 1] if j > 0 else 1e30, t[i, j + 1] if j < nz - 1 else 1e30)
                    fl = f[i, j]
                    tn = min(a + hs * fl, b + hz * fl)
                    if abs(a - b) < 1e29:
                        B = -2.0 * (a / hs**2 + b / hz**2)
                        C = a * a / hs**2 + b * b / hz**2 - fl * fl
                        disc = B * B - 4 * A * C
                        if disc >= 0:
                            tq = (-B + np.sqrt(disc)) / (2 * A)
                            if tq > max(a, b):
                                tn = min(tn, tq)
                    if tn < t[i, j]:
                        change = max(change, t[i, j] - tn)
                        t[i, j] = tn
                    j += dj
                i += di
        if change < 1e-7:
            break
    return t


def rupture_time(s, z, hypo_s, hypo_z, v):
    """first-arrival time (s) from the hypocenter through the speed field v[ns,nz] (fast sweeping, 1st order).
    Uniform v: exact analytic distance/v."""
    S, Z = np.meshgrid(s, z, indexing='ij')
    if np.ptp(v) < 1e-9:
        return np.hypot(S - hypo_s, Z - hypo_z) / v.flat[0]
    hs, hz = s[1] - s[0], z[1] - z[0]
    t = np.full(v.shape, 1e30)
    d = np.hypot(S - hypo_s, Z - hypo_z)
    near = d <= 1.5 * max(hs, hz)                        # seed the neighborhood of the hypocenter with straight-line times
    t[near] = d[near] / v[near]
    th = _sweep(t, 1.0 / v, hs, hz, 200)
    # remove the grid anisotropy of 1st-order fast sweeping: same solve for uniform v, rescale to the exact distance/v
    vm = float(np.mean(v))
    tu = np.full(v.shape, 1e30); tu[near] = d[near] / vm
    tu = _sweep(tu, np.full(v.shape, 1.0 / vm), hs, hz, 200)
    ex = d / vm
    corr = np.where(tu > 1e-6, ex / np.maximum(tu, 1e-6), 1.0)
    return th * corr


def stochastic_scenario(s, z, L, W, hypo_s, hypo_z, Dmean=2.0, Dmax=None, vr=2700.0, vr_std=0.0, vr_seg=1000.0,
                        vr_seg_z=0.0, vr_min=1000.0, vr_max=3300.0, vr_step=False, rise=1.5,
                        taper_s=3000.0, taper_z=4000.0, slip_sigma=0.0, corr_s=3000.0, corr_z=2000.0,
                        ssd_pct=None, ssd_depth=1000.0, ssd_smooth=0.0, shallow_frac=1.0, shallow_z=3000.0, seed=1):
    """returns D[ns,nz], t0, T, v  (slip m, rupture time s, rise time s, rupture speed m/s)."""
    rng = np.random.default_rng(seed)
    rng_v = np.random.default_rng(seed + 1000003)
    ds, dz = s[1] - s[0], z[1] - z[0]
    S, Z = np.meshgrid(s, z, indexing='ij')
    ws, wz = taper_weights(S, Z, L, W, taper_s, taper_z)
    D = ws * wz
    if shallow_frac < 1.0:                               # legacy linear surface ramp (use ssd_pct instead for a box deficit)
        D = D * (shallow_frac + (1 - shallow_frac) * np.clip(Z / shallow_z, 0, 1))
    if slip_sigma > 0:
        n = correlated_noise(S.shape, ds, dz, corr_s, corr_z, rng)
        D = D * np.exp(slip_sigma * n - 0.5 * slip_sigma**2)      # log-normal, mean-preserving
    if ssd_pct is not None:
        D = apply_shallow_deficit(D, z, ssd_pct, ssd_depth, ssd_smooth)
    D = D * (Dmax / D.max() if Dmax else Dmean / D.mean())
    v = speed_field(s, z, L, W, vr, vr_std, vr_seg, vr_seg_z, vr_min, vr_max, rng_v, vr_step)
    t0 = rupture_time(s, z, hypo_s, hypo_z, v)
    return D, t0, np.full_like(D, rise), v
