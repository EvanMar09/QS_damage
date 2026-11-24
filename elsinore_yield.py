#!/usr/bin/env python3
"""
elsinore_yield.py - kinematic screening of off-fault Drucker-Prager yielding on the Elsinore fault.

  geometry  : smoothed trace read from the gmsh .geo (Point(1001..)=...), vertical fault, local mesh frame
  source    : slip D(s,z), rupture time t0(s,z), rise time T(s,z)  [synthetic default, or --slip-npz,
              or SeisSol fault output via --seissol-fault]
  medium    : homogeneous full space (alpha,beta,rho)
  stress    : initial tensor (brief section 5) + kinematic stress change, tested against Drucker-Prager
  output    : results npz, ParaView .vtk, summary (plots: plot_results.py)
"""
import argparse, re, time, json
import os
import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator
import kinstress as ks
import scenario_stoch as sst

G = 9.81

# ------------------------------------------------------------------ geometry
def parse_geo_trace(path):
    pts = {}
    for m in re.finditer(r'Point\((\d+)\)\s*=\s*\{\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)', open(path).read()):
        idx = int(m.group(1))
        if idx >= 1000:                                   # fault-trace points (domain-box corners are 1..8)
            pts[idx] = (float(m.group(2)), float(m.group(3)))
    xy = np.array([pts[k] for k in sorted(pts)])
    return xy


class Trace:
    """Arc-length parametrized cubic spline through the trace points (local frame, meters)."""
    def __init__(self, xy):
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        self.L = d[-1]
        self.cs = CubicSpline(d, xy, bc_type='natural')

    def pos(self, s):
        return self.cs(np.clip(s, 0, self.L))

    def tang(self, s):
        t = self.cs(np.clip(s, 0, self.L), 1)
        return t / np.hypot(t[..., 0], t[..., 1])[..., None]

    def normal(self, s):                                   # left normal = z x t
        t = self.tang(s)
        return np.stack([-t[..., 1], t[..., 0]], -1)

    def project(self, xy, n=20001):
        ss = np.linspace(0, self.L, n)
        p = self.pos(ss)
        return np.array([ss[np.argmin(np.hypot(*(p - q).T))] for q in np.atleast_2d(xy)])


# ------------------------------------------------------------------ material / initial stress (brief s.4-5)
LAYERS_TOP = np.array([0, 1000, 3000, 6000, 10000.])       # layer tops (m)
LAYERS_RHO = np.array([2300, 2500, 2600, 2700, 2750.])
STRESS_DEFAULTS = dict(coef=0.25, clamp=5000.0, shear_ratio=13.4347 / 36.7875)


def parse_layers(txt):
    """'0:2300,1000:2500,3000:2600' -> (tops m, densities kg/m3)"""
    pairs = [t.split(':') for t in txt.split(',')]
    return np.array([float(a) for a, b in pairs]), np.array([float(b) for a, b in pairs])


_FACTORY = dict(defaults=dict(STRESS_DEFAULTS), top=LAYERS_TOP.copy(), rho=LAYERS_RHO.copy())


def reset_stress_params():
    """Back to the factory initial stress (0.25 g, 5 km clamp, built-in layers, no yaml); stops one run's settings leaking into the next
    when several runs are made in the same Python session (Spyder)."""
    global LAYERS_TOP, LAYERS_RHO, _YAML_STRESS
    STRESS_DEFAULTS.update(_FACTORY['defaults']); LAYERS_TOP, LAYERS_RHO = _FACTORY['top'].copy(), _FACTORY['rho'].copy()
    _YAML_STRESS = None


def set_stress_params(coef=None, clamp=None, shear_ratio=None, layers=None):
    """Change the built-in (formula) initial stress: lithostatic coefficient, depth where the mean stress stops growing,
    deviatoric/mean ratio, density layers ('top:density,...')."""
    global LAYERS_TOP, LAYERS_RHO
    if coef is not None: STRESS_DEFAULTS['coef'] = float(coef)
    if clamp is not None: STRESS_DEFAULTS['clamp'] = float(clamp)
    if shear_ratio is not None: STRESS_DEFAULTS['shear_ratio'] = float(shear_ratio)
    if layers: LAYERS_TOP, LAYERS_RHO = parse_layers(layers) if isinstance(layers, str) else layers


def mean_stress(depth, coef=None, clamp=None):
    """sigma_mean = coef*g*int_0^min(depth,clamp) rho dz   (positive magnitude, Pa)."""
    coef = STRESS_DEFAULTS['coef'] if coef is None else coef
    clamp = STRESS_DEFAULTS['clamp'] if clamp is None else clamp
    zz = np.minimum(np.asarray(depth, float), clamp)
    out = np.zeros_like(zz)
    for top, nxt, rho in zip(LAYERS_TOP, np.r_[LAYERS_TOP[1:], 1e9], LAYERS_RHO):
        out += rho * np.clip(zz - top, 0, nxt - top)
    return coef * G * out


_YAML_STRESS = None          # set by use_seissol_yaml(): function depth -> (..., 6)


def use_seissol_yaml(path):
    """Take the initial (regional) stress from a SeisSol easi yaml: the Lua function in the file is run (needs `pip install lupa`)
    and its s_xx, s_yy, s_zz, s_xy, s_yz, s_xz are used at every depth.  Only depth dependence is supported (the stress must not
    depend on x, y); the fault-only entries (mu_s, mu_d, cohesion, nucleation) are not used."""
    global _YAML_STRESS
    import textwrap
    try:
        import lupa
    except ImportError:
        raise SystemExit('reading the SeisSol yaml needs the lupa package:  pip install lupa')
    lines = open(path).read().splitlines()
    k = next((i for i, l in enumerate(lines) if l.strip().startswith('function:')), None)
    if k is None:
        raise SystemExit(f'{path}: no "function: |" Lua block found')
    ind0 = len(lines[k]) - len(lines[k].lstrip())
    body = []
    for l in lines[k + 1:]:
        if l.strip() and len(l) - len(l.lstrip()) <= ind0:
            break
        body.append(l)
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute(textwrap.dedent('\n'.join(body)))
    fn = lua.globals().f
    keys = ['s_xx', 's_yy', 's_zz', 's_xy', 's_yz', 's_xz']

    def at(x, y, z):
        r = fn(lua.table_from({'x': float(x), 'y': float(y), 'z': float(z)}))
        return np.array([r[kk] for kk in keys], float)

    for dz in (0.0, 500.0, 2500.0, 8000.0, 14000.0):           # the stress must not depend on horizontal position
        a, b = at(0.0, 0.0, -dz), at(9.0e3, -6.0e3, -dz)
        if np.abs(a - b).max() > 1e-6 * max(np.abs(a).max(), 1.0):
            raise SystemExit('the yaml stress depends on x or y; only depth-dependent stress is supported')

    def f(depth):
        depth = np.asarray(depth, float)
        u, inv = np.unique(depth.ravel(), return_inverse=True)
        tab = np.array([at(0.0, 0.0, -zz) for zz in u])
        return tab[inv].reshape(depth.shape + (6,))

    _YAML_STRESS = f
    return f


def restore_stress_from_params(p):
    """Re-activate the initial stress of a saved run (its params dict): formula parameters, or the SeisSol yaml if it used one."""
    reset_stress_params()
    set_stress_params(p.get('litho_coef'), p.get('clamp_depth'), p.get('shear_ratio'), p.get('layers'))
    y = p.get('seissol_yaml')
    if y:
        try: use_seissol_yaml(y)
        except (OSError, SystemExit) as e: print(f'WARNING: run used the yaml stress {y} but it cannot be loaded now ({e}); using the built-in stress')


def initial_stress(depth, shmax_az=350.0, shear_ratio=None, coef=None, clamp=None):
    """Initial tensor, compression negative, order xx yy zz xy yz xz.  depth (m, positive down).
    If use_seissol_yaml() was called, the stress comes from that yaml instead."""
    if _YAML_STRESS is not None:
        return _YAML_STRESS(depth)
    shear_ratio = STRESS_DEFAULTS['shear_ratio'] if shear_ratio is None else shear_ratio
    sm = mean_stress(depth, coef, clamp)
    sd = shear_ratio * sm
    SH, Sh = sm + sd, sm - sd
    phi = np.radians(90.0 - shmax_az)
    c, s = np.cos(phi), np.sin(phi)
    out = np.zeros(depth.shape + (6,))
    out[..., 0] = -(SH * c * c + Sh * s * s)
    out[..., 1] = -(SH * s * s + Sh * c * c)
    out[..., 2] = -sm
    out[..., 3] = -(SH - Sh) * s * c
    return out


# ------------------------------------------------------------------ scenario: slip / rupture time / rise time
def synthetic_scenario(s, depth, L, W, hypo_s, hypo_depth, Dmean=2.0, vr=2770.0, rise=1.5,
                       taper_s=3000.0, taper_z=4000.0, shallow_frac=0.5, shallow_z=3000.0,
                       rough=0.0, seed=1):
    """Smooth tapered slip + circular rupture front from the hypocenter on the (developable) fault plane."""
    S, Z = np.meshgrid(s, depth, indexing='ij')
    ws, wz = sst.taper_weights(S, Z, L, W, taper_s, taper_z)
    sh = shallow_frac + (1 - shallow_frac) * np.clip(Z / shallow_z, 0, 1)
    D = ws * wz * sh
    if rough > 0:
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(seed)
        n = rng.standard_normal(S.shape)
        dx = s[1] - s[0]
        n = gaussian_filter(n, 3000.0 / dx, mode='wrap')
        n /= n.std()
        D = D * np.exp(rough * n)
    D *= Dmean / D.mean()
    t0 = np.hypot(S - hypo_s, Z - hypo_depth) / vr
    T = np.full_like(D, rise)
    return D, t0, T


def load_npz_scenario(path, s, depth):
    """npz with 1-D axes s (m along trace from SE end) and depth (m) and 2-D arrays slip, rt[, rise] shaped (ns,nd)."""
    d = np.load(path)
    out = []
    for key, default in (('slip', None), ('rt', None), ('rise', 1.5)):
        if key in d:
            f = RegularGridInterpolator((d['s'], d['depth']), d[key], bounds_error=False, fill_value=None)
            S, Z = np.meshgrid(s, depth, indexing='ij')
            out.append(f(np.stack([S.ravel(), Z.ravel()], -1)).reshape(S.shape))
        else:
            out.append(np.full((len(s), len(depth)), default))
    return out


def load_seissol_fault(xdmf, trace, s, depth, idt=-1):
    """Interpolate SeisSol fault output (ASl, RT [, PSR]) onto the (s,depth) source grid.  (needs seissolxdmf)"""
    import seissolxdmf
    from scipy.interpolate import griddata
    sx = seissolxdmf.seissolxdmf(xdmf)
    xyz = sx.ReadGeometry()
    con = sx.ReadConnect()
    cen = xyz[con].mean(axis=1)
    nt = sx.ReadNdt()
    asl = sx.ReadData('ASl', nt - 1 if idt < 0 else idt)
    rt = sx.ReadData('RT', nt - 1 if idt < 0 else idt)
    cs = trace.project(cen[:, :2])
    pts = np.stack([cs, -cen[:, 2]], -1)
    S, Z = np.meshgrid(s, depth, indexing='ij')
    q = np.stack([S.ravel(), Z.ravel()], -1)
    def interp(v):
        a = griddata(pts, v, q, method='linear')
        b = griddata(pts, v, q, method='nearest')
        return np.where(np.isnan(a), b, a).reshape(S.shape)
    D = interp(asl)
    t0 = interp(rt)
    T = np.full_like(D, 1.5)
    if 'PSR' in sx.ReadAvailableDataFields():             # raised cosine: peak rate = pi*D/(2T)
        psr = interp(sx.ReadData('PSR', nt - 1 if idt < 0 else idt))
        T = np.clip(np.pi * D / (2 * np.maximum(psr, 1e-6)), 0.3, 6.0)
    return D, t0, T


# ------------------------------------------------------------------ field-measured surface slip
UTM_ORIGIN = (578629.759, 3634947.523)          # local = UTM11N - origin


def parse_field_slip(path, trace, bin_km=1.0):
    """Read the Google-Earth export (lat/lon + 'name' like '1.7m RL').  Keeps right-lateral offsets only
    (drops vertical-only and pure dip-slip entries), projects onto the trace and takes the median per bin.
    Returns raw (s, slip, is_min) and binned (s_bin, slip_bin)."""
    from pyproj import Transformer
    tf = Transformer.from_crs('EPSG:4326', 'EPSG:32611', always_xy=True)
    raw = []
    for ln in open(path):
        p = ln.rstrip('\n').split('\t')
        if len(p) < 9 or p[0] != 'W':
            continue
        name = p[8]
        m = re.match(r'\s*([\d.]+)\s*m', name)
        if not m or 'dip slip' in name or re.search(r'\bV\b', name):
            continue
        e, n = tf.transform(float(p[2]), float(p[1]))
        sp = trace.project([[e - UTM_ORIGIN[0], n - UTM_ORIGIN[1]]])[0]
        raw.append((sp, float(m.group(1)), '(min)' in name))
    raw = np.array(raw, dtype=float)
    edges = np.arange(0, trace.L + bin_km * 1e3, bin_km * 1e3)
    sb, vb = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (raw[:, 0] >= lo) & (raw[:, 0] < hi)
        if k.any():
            sb.append(raw[k, 0].mean()); vb.append(np.median(raw[k, 1]))
    return raw, np.array(sb), np.array(vb)


def field_scenario(s, depth, L, W, hypo_s, hypo_depth, prof_s, prof_v, extrap='hold', smooth=1500.0,
                   vr=2770.0, rise=1.5, taper_s=3000.0, taper_z=4000.0, shallow_frac=1.0, shallow_z=3000.0,
                   scale=1.0, depth_scale=1.0, scale_depth=1000.0, scale_smooth=0.0,
                   noise_sigma=0.0, noise_depth=3000.0, corr_s=3000.0, corr_z=2000.0, seed=1):
    """D(s,z) = field surface-slip profile (smoothed, extrapolated) x depth scaling x end taper x optional random field.

    depth scaling: slip at the surface = the field profile; slip below `scale_depth` = depth_scale x the surface slip.
      scale_smooth 0 -> abrupt step at scale_depth;  1 -> cosine change spread over the whole 0..scale_depth layer.
    random field (noise_sigma > 0): smooth log-normal multiplier (mean 1, std ~noise_sigma, correlation lengths corr_s/corr_z).
      Its amplitude grows from 0 at the surface to full at `noise_depth` so the surface keeps honoring the field data
      (noise_depth <= 0 applies it at all depths)."""
    from scipy.ndimage import gaussian_filter1d
    from scenario_stoch import depth_ramp, correlated_noise
    ds = s[1] - s[0]
    if extrap == 'hold':
        prof = np.interp(s, prof_s, prof_v)
    else:                                              # 'taper': ramp to zero within 3 km beyond last datum
        prof = np.interp(s, prof_s, prof_v, left=0.0, right=0.0)
        for edge, sign in ((prof_s[0], -1), (prof_s[-1], 1)):
            r = np.clip(1 - sign * (s - edge) / 3000.0, 0, 1)
            prof = np.where(sign * (s - edge) > 0, np.interp(s, prof_s, prof_v) * r, prof)
    prof = gaussian_filter1d(prof, smooth / ds, mode='nearest')
    S, Z = np.meshgrid(s, depth, indexing='ij')
    ws, wz = sst.taper_weights(S, Z, L, W, taper_s, taper_z)
    sh = shallow_frac + (1 - shallow_frac) * np.clip(Z / shallow_z, 0, 1)
    gz = depth_scale * depth_ramp(depth, scale_depth, 1.0 / depth_scale, scale_smooth) if depth_scale != 1.0 else np.ones(len(depth))
    D = scale * prof[:, None] * gz[None, :] * ws * wz * sh
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        n = correlated_noise(S.shape, ds, depth[1] - depth[0], corr_s, corr_z, rng)
        sig = noise_sigma * (np.clip(depth / noise_depth, 0, 1) if noise_depth > 0 else np.ones(len(depth)))[None, :]
        D = D * np.exp(sig * n - 0.5 * sig ** 2)
    t0 = np.hypot(S - hypo_s, Z - hypo_depth) / vr
    return D, t0, np.full_like(D, rise)


def select_transects(spec, n):
    """0-based indices of the transects to use out of n.  spec: 'all' (or '' / None), 'every:5' (1, 6, 11, ...),
    'n:10' (10 evenly spaced, first and last included) or an explicit 1-based list '1,6,11,21'."""
    spec = '' if spec is None else str(spec).strip().lower()
    if spec in ('', 'all'):
        return np.arange(n)
    if spec.startswith('every'):
        k = int(spec.split(':')[1] if ':' in spec else spec.split('=')[1]); idx = np.arange(0, n, max(k, 1))
    elif spec.startswith('n:') or spec.startswith('n='):
        m = int(spec[2:]); idx = np.unique(np.round(np.linspace(0, n - 1, min(max(m, 1), n))).astype(int))
    else:
        idx = np.array([int(v) - 1 for v in spec.split(',') if v.strip()])
    if idx.size == 0 or idx.min() < 0 or idx.max() >= n:
        raise SystemExit(f'--transect-subset {spec!r}: transect numbers must lie between 1 and {n}')
    return idx


def read_transects(path, trace):
    """receivers_transects_perp*.dat: 50 transects x 10 stations (x y z), offsets -1000..+1000 (- = NE, + = SW)."""
    a = np.loadtxt(path).reshape(-1, 10, 3)
    cen = 0.5 * (a[:, 4, :2] + a[:, 5, :2])
    nrm = a[:, 9, :2] - a[:, 0, :2]
    nrm /= np.hypot(*nrm.T)[:, None]
    return cen, nrm, trace.project(cen)


# ------------------------------------------------------------------ source arrays (two-level)
def build_sources(trace, s, depth, D, t0, T, mu, bs=5, bz=5):
    """fine[nb,nc,8], coarse[nb,8]; columns x y z tx ty M0 t0 T.  Blocks of bs x bz cells."""
    h = s[1] - s[0]
    ns, nz = D.shape
    nsp, nzp = -(-ns // bs) * bs, -(-nz // bz) * bz
    def pad(a, fill=0.0):
        out = np.full((nsp, nzp), fill)
        out[:ns, :nz] = a
        return out
    sp_ = np.r_[s, s[-1] + h * np.arange(1, nsp - ns + 1)]
    zp_ = np.r_[depth, depth[-1] + h * np.arange(1, nzp - nz + 1)]
    P = trace.pos(sp_)
    Tn = trace.tang(sp_)
    Dp, t0p, Tp = pad(D), pad(t0), pad(T, 1.0)
    M0 = mu * h * h * Dp
    fine = np.zeros((nsp, nzp, 8))
    fine[..., 0] = P[:, 0][:, None]
    fine[..., 1] = P[:, 1][:, None]
    fine[..., 2] = -zp_[None, :]
    fine[..., 3] = Tn[:, 0][:, None]
    fine[..., 4] = Tn[:, 1][:, None]
    fine[..., 5] = M0
    fine[..., 6] = t0p
    fine[..., 7] = Tp
    def blocks(a):                                        # (nsp,nzp[,k]) -> (nb, bs*bz[,k])
        sh = a.shape[2:] if a.ndim > 2 else ()
        a = a.reshape(nsp // bs, bs, nzp // bz, bz, *sh)
        a = np.moveaxis(a, 2, 1)
        return a.reshape(-1, bs * bz, *sh)
    fb = blocks(fine)                                     # (nb, nc, 8)
    w = fb[..., 5]
    ws = w.sum(1)
    coarse = np.zeros((fb.shape[0], 8))
    ok = ws > 0
    wn = np.where(ok[:, None], w / np.where(ws > 0, ws, 1)[:, None], 0)
    for c in (0, 1, 2, 6, 7):
        coarse[:, c] = (wn * fb[..., c]).sum(1)
    tt = np.stack([(wn * fb[..., 3]).sum(1), (wn * fb[..., 4]).sum(1)], -1)
    tt /= np.maximum(np.hypot(*tt.T), 1e-12)[:, None]
    coarse[:, 3:5] = tt
    coarse[:, 5] = ws
    big = np.where(w > 0, fb[..., 6], np.nan)
    spread = np.nan_to_num(np.nanmax(np.where(np.isnan(big), -1e30, big), 1) - np.nanmin(np.where(np.isnan(big), 1e30, big), 1))
    spread = np.where(ok, spread, 0.0)
    coarse[:, 7] += 0.5 * spread                          # block rise time absorbs rupture-time spread
    # coarse center must be an unweighted block center for the near/far test
    cen = fb[..., :3].mean(1)
    coarse[:, :3] = np.where(ok[:, None], coarse[:, :3], cen)
    return np.ascontiguousarray(fb), np.ascontiguousarray(coarse)


# ------------------------------------------------------------------ evaluation grid
def build_receivers(trace, s_rec, off_rec, depth_rec):
    P = trace.pos(s_rec)
    N = trace.normal(s_rec)
    S, O, Z = np.meshgrid(np.arange(len(s_rec)), off_rec, depth_rec, indexing='ij')
    xy = P[S] + O[..., None] * N[S]
    rec = np.stack([xy[..., 0], xy[..., 1], -Z], -1)
    return rec.reshape(-1, 3), rec.shape[:3]


# ------------------------------------------------------------------ static stress drop on the fault
def static_stress_drop(trace, fine, coarse, med, h, W, a_eps, r_near, s_map, z_map, delta, rup_end=30.0, shmax=350.0):
    """Static shear-stress change resolved on the fault (strike direction, fault-normal traction component),
    evaluated at +-delta from the fault and averaged over both sides (delta ~ 1.5 h keeps the point-source
    discretization error to a few %).  Returns dtau[ns,nz] (Pa, negative = stress drop) and the initial tau0, sigma_n0."""
    P = trace.pos(s_map); Tg = trace.tang(s_map); N = trace.normal(s_map)
    S, Z = np.meshgrid(np.arange(len(s_map)), z_map, indexing='ij')
    out = []
    for sign in (-1.0, 1.0):
        xy = P[S] + sign * delta * N[S]
        rec = np.stack([xy[..., 0], xy[..., 1], -Z], -1).reshape(-1, 3)
        dt = 1.0
        nt = int(rup_end + trace.L / 3000.0 + 10)
        St = ks.stress_history(rec, fine, coarse, med, dt, nt, r_near, 1e12, a_eps)[:, -1, :]
        TG = Tg[S].reshape(-1, 2); NM = N[S].reshape(-1, 2)
        dt_ = TG[:, 0] * NM[:, 0] * St[:, 0] + TG[:, 1] * NM[:, 1] * St[:, 1] + (TG[:, 0] * NM[:, 1] + TG[:, 1] * NM[:, 0]) * St[:, 3]
        out.append(dt_.reshape(S.shape))
    dtau = 0.5 * (out[0] + out[1])
    i0 = initial_stress(np.broadcast_to(z_map[None, :], dtau.shape), shmax)
    TGm = Tg[S]; NMm = N[S]
    tau0 = TGm[..., 0] * NMm[..., 0] * i0[..., 0] + TGm[..., 1] * NMm[..., 1] * i0[..., 1] + (TGm[..., 0] * NMm[..., 1] + TGm[..., 1] * NMm[..., 0]) * i0[..., 3]
    sn0 = -(NMm[..., 0] ** 2 * i0[..., 0] + NMm[..., 1] ** 2 * i0[..., 1] + 2 * NMm[..., 0] * NMm[..., 1] * i0[..., 3])
    return dtau, tau0, sn0


def stress_drop_summary(D, dtau_on_src_grid, h, mu, thresh=0.1):
    """Stress-drop statistics (positive numbers = drop).  D, dtau on the same (s,z) grid."""
    m = D > thresh * D.max()
    A = m.sum() * h * h
    M0 = mu * (D * h * h).sum()
    drop = -dtau_on_src_grid
    a_eq = np.sqrt(A / np.pi)
    return dict(area_km2=A / 1e6, M0=M0, mean_slip=D[m].mean(),
                mean_drop_area=drop[m].mean(), mean_drop_slipweighted=(drop[m] * D[m]).sum() / D[m].sum(),
                eshelby_drop=7.0 / 16.0 * M0 / a_eq ** 3, a_eq=a_eq)


def rebuild_from_npz(d, geo):
    """Recreate trace, medium and the two-level sources from a saved run (slip/rt/rise grids)."""
    import json
    p = json.loads(str(d['params']))
    trace = Trace(parse_geo_trace(geo))
    med = ks.medium(p['alpha'], p['beta'], p['rho'])
    fine, coarse = build_sources(trace, d['s_src'], d['z_src'], d['slip'], d['rt'], d['rise'], med[4], p['blocks'], p['blocks'])
    h = float(d['s_src'][1] - d['s_src'][0])
    return trace, med, fine, coarse, h, p


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--geo', default='Elsinore_simple30m_smooth1.geo')
    ap.add_argument('--out', default='run_default')
    ap.add_argument('--h', type=float, default=100.0, help='source subfault size (m); for --solver ar keep the smallest offset >= 3h')
    ap.add_argument('--depth', type=float, default=15000.0, help='fault width (m)')
    ap.add_argument('--hypo', default='M', help='N, M, S or "x,y" in local meters')
    ap.add_argument('--hypo-depth', type=float, default=8000.0)
    ap.add_argument('--slip', type=float, default=2.0, help='mean slip (m), synthetic scenario')
    ap.add_argument('--max-slip', type=float, default=None, help='peak slip (m), synthetic scenario; overrides --slip')
    ap.add_argument('--vr', type=float, default=2770.0, help='rupture speed (m/s)')
    ap.add_argument('--rise', type=float, default=1.5, help='rise time (s) (--rise-mode constant)')
    ap.add_argument('--rise-mode', default='constant', choices=['constant', 'gp'], help="'constant' = --rise everywhere; 'gp' = Graves & Pitarka (2016): rise time ~ sqrt(local slip), x2 in the shallow zone, mean from M0")
    ap.add_argument('--rise-mean', type=float, default=None, help="gp mode: fault-average rise time (s); default = self-similar 1.6e-9 M0^(1/3) (M0 in dyne-cm)")
    ap.add_argument('--rise-eta', type=float, default=0.8, help='gp mode: correlation of the rise-time field with the slip (1 = tau ~ sqrt(slip) exactly, 0 = independent; GP2016 used 0.8)')
    ap.add_argument('--no-rise-depth-factor', action='store_true', help='gp mode: no doubling of the rise time above 5 km (and below 18 km)')
    ap.add_argument('--shallow-frac', type=float, default=None, help='slip at surface relative to >=3 km depth (default 0.5 synthetic, 1.0 field)')
    ap.add_argument('--rough', type=float, default=0.0, help='(old name for --slip-sigma)')
    ap.add_argument('--slip-sigma', type=float, default=None, help='log-normal std of the stochastic slip field (0 = smooth taper only)')
    ap.add_argument('--corr-s', type=float, default=3000.0, help='slip correlation length along strike (m)')
    ap.add_argument('--corr-z', type=float, default=2000.0, help='slip correlation length down dip (m)')
    ap.add_argument('--seed', type=int, default=1, help='random seed (slip field and rupture speed)')
    ap.add_argument('--ssd', type=float, default=None, help='shallow slip deficit: slip in the upper --ssd-depth is this %% of the mean slip below it')
    ap.add_argument('--ssd-depth', type=float, default=1000.0, help='depth of the shallow-deficit layer (m)')
    ap.add_argument('--ssd-smooth', type=float, default=0.0, help='0..1: 0 = abrupt step at --ssd-depth, 1 = slip rises smoothly from the surface value to 100%% across the whole layer')
    ap.add_argument('--vr-std', type=float, default=0.0, help='std of rupture speed (m/s); 0 = uniform --vr')
    ap.add_argument('--vr-seg', type=float, default=1000.0, help='along-strike spacing of rupture-speed samples (m)')
    ap.add_argument('--vr-seg-z', type=float, default=0.0, help='down-dip spacing of speed samples (m); 0 = same speed at all depths')
    ap.add_argument('--vr-min', type=float, default=1000.0); ap.add_argument('--vr-max', type=float, default=3300.0)
    ap.add_argument('--vr-step', action='store_true', help='piecewise-constant speed between samples (default: linear)')
    ap.add_argument('--field-slip', default=None, help='lat/lon txt of field offsets (RL values in the name column)')
    ap.add_argument('--field-extrap', default='hold', choices=['hold', 'taper'], help='slip beyond the data range')
    ap.add_argument('--field-smooth', type=float, default=1500.0, help='along-strike smoothing of field profile (m)')
    ap.add_argument('--field-scale', type=float, default=1.0)
    ap.add_argument('--depth-scale', type=float, default=1.0, help='field slip: slip below --depth-scale-depth = this x the surface (field) slip')
    ap.add_argument('--depth-scale-depth', type=float, default=1000.0, help='depth (m) at which the depth scaling is complete')
    ap.add_argument('--depth-scale-smooth', type=float, default=0.0, help='0..1: 0 = abrupt step at --depth-scale-depth, 1 = gradual over the whole layer above it')
    ap.add_argument('--field-noise', type=float, default=0.0, help='field slip: log-normal std of a smooth random multiplier (0 = off); uses --corr-s/--corr-z/--seed')
    ap.add_argument('--field-noise-depth', type=float, default=3000.0, help='depth (m) over which the random multiplier grows from 0 (surface keeps the field data) to full; 0 = all depths')
    ap.add_argument('--taper-s', type=float, default=3000.0, help='slip taper length at both ends along strike (m); 0 = no taper (full slip up to the fault ends)')
    ap.add_argument('--taper-z', type=float, default=4000.0, help='slip taper length above the fault base (m); 0 = no taper (full slip down to the base)')
    ap.add_argument('--transect-subset', default='all', help="use only some of the transects: 'all', 'every:5', 'n:10' (evenly spaced) or a 1-based list '1,6,11'")
    ap.add_argument('--transects', default=None, help='receivers_transects_perp_*.dat: evaluate on these transects')
    ap.add_argument('--slip-npz', default=None)
    ap.add_argument('--seissol-fault', default=None)
    ap.add_argument('--alpha', type=float, default=6000.0)
    ap.add_argument('--beta', type=float, default=3464.0)
    ap.add_argument('--rho', type=float, default=2700.0)
    ap.add_argument('--coh', type=float, default=5e6, help='cohesion (Pa)')
    ap.add_argument('--bulk-friction', type=float, default=0.65, help='tan(phi)  [check SeisSol mapping!]')
    ap.add_argument('--shmax', type=float, default=350.0, help='SHmax azimuth for the built-in initial stress (ignored with --seissol-yaml)')
    ap.add_argument('--litho-coef', type=float, default=0.25, help='built-in stress: mean stress = coef * g * integral(rho dz)')
    ap.add_argument('--clamp-depth', type=float, default=5000.0, help='built-in stress: depth (m) below which the mean stress stops growing')
    ap.add_argument('--shear-ratio', type=float, default=13.4347 / 36.7875, help='built-in stress: deviatoric stress / mean stress')
    ap.add_argument('--layers', default='0:2300,1000:2500,3000:2600,6000:2700,10000:2750', help="built-in stress density layers 'top_m:rho,...'")
    ap.add_argument('--seissol-yaml', default=None, help='SeisSol easi yaml (Lua function) to take the initial stress from (needs lupa)')
    ap.add_argument('--solver', default='ar', choices=['ar', 'qs-full', 'qs-half'],
                    help='ar = dynamic full-space (Aki & Richards); qs-full / qs-half = quasi-static dislocation patches in a full space / half-space (free surface)')
    ap.add_argument('--no-kernel-cache', action='store_true', help='quasi-static solvers: do not save/reuse the geometry kernel')
    ap.add_argument('--kernel-cache-dir', default=None, help='folder for the kernel cache (default: _kernel_cache next to --out)')
    ap.add_argument('--kernel-cache-max-gb', type=float, default=3.0, help='skip caching if the kernel would be larger than this')
    ap.add_argument('--tv', type=float, default=0.3, help='SeisSol viscoplastic relaxation time Tv (s): the sustained-yield criterion is F > 0 for at least ceil(Tv/dt) consecutive time steps (0 = same as peak F > 0)')
    ap.add_argument('--qs-theta', type=float, default=2.5, help='quasi-static: block-merging distance in block sizes')
    ap.add_argument('--ds', type=float, default=300.0, help='along-strike receiver spacing (m)')
    ap.add_argument('--dz', type=float, default=300.0, help='depth receiver spacing (m)')
    ap.add_argument('--offsets', default='150,300,500,800', help='|offset| list (m), both sides are computed')
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--tmax', type=float, default=45.0)
    ap.add_argument('--r-near', type=float, default=None, help='fine/coarse switch distance (m); default 3*blocks*h')
    ap.add_argument('--rcut', type=float, default=1e9)
    ap.add_argument('--eps-frac', type=float, default=0.5, help='near-field softening = eps_frac*h')
    ap.add_argument('--blocks', type=int, default=5)
    ap.add_argument('--save-times', default='', help='comma list of times (s) at which F(t) is stored for plotting time slices')
    a = ap.parse_args(argv)

    t_start = time.time()
    if os.path.dirname(a.out):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
    trace = Trace(parse_geo_trace(a.geo))
    L, W, h = trace.L, a.depth, a.h
    med = ks.medium(a.alpha, a.beta, a.rho)
    print(f'trace length {L/1e3:.2f} km, width {W/1e3:.1f} km, h={h:.0f} m, mu={med[4]/1e9:.1f} GPa')

    ns, nz = int(round(L / h)), int(round(W / h))
    s_src = (np.arange(ns) + 0.5) * (L / ns)
    z_src = (np.arange(nz) + 0.5) * (W / nz)
    h = L / ns                                            # tiny adjustment so cells tile the trace exactly
    HYPO = {'N': (-5734.5, 2609.9), 'M': (-2114.8, -834.7), 'S': (8593.1, -5400.1)}
    hx, hy = HYPO[a.hypo] if a.hypo in HYPO else tuple(float(v) for v in a.hypo.split(','))
    hs = float(trace.project([[hx, hy]])[0])
    print(f'hypocenter {a.hypo}: s = {hs/1e3:.2f} km from SE end, depth {a.hypo_depth/1e3:.1f} km')

    field_raw = None; vgrid = None
    if a.field_slip:
        field_raw, ps, pv = parse_field_slip(a.field_slip, trace)
        print(f'field slip: {len(field_raw)} RL measurements, s = {field_raw[:,0].min()/1e3:.1f}-{field_raw[:,0].max()/1e3:.1f} km, '
              f'binned median range {pv.min():.2f}-{pv.max():.2f} m')
        D, t0, T = field_scenario(s_src, z_src, L, W, hs, a.hypo_depth, ps, pv, a.field_extrap, a.field_smooth,
                                  a.vr, a.rise, taper_s=a.taper_s, taper_z=a.taper_z, shallow_frac=1.0 if a.shallow_frac is None else a.shallow_frac, scale=a.field_scale,
                                  depth_scale=a.depth_scale, scale_depth=a.depth_scale_depth, scale_smooth=a.depth_scale_smooth,
                                  noise_sigma=a.field_noise, noise_depth=a.field_noise_depth, corr_s=a.corr_s, corr_z=a.corr_z, seed=a.seed)
        if a.depth_scale != 1.0 or a.field_noise > 0:
            print(f'field slip depth scaling: x{a.depth_scale:g} below {a.depth_scale_depth:g} m (smooth {a.depth_scale_smooth:g})'
                  + (f', random multiplier sigma {a.field_noise:g}, seed {a.seed}' if a.field_noise > 0 else ''))
    elif a.seissol_fault:
        D, t0, T = load_seissol_fault(a.seissol_fault, trace, s_src, z_src)
    elif a.slip_npz:
        D, t0, T = load_npz_scenario(a.slip_npz, s_src, z_src)
    else:
        sig = a.slip_sigma if a.slip_sigma is not None else a.rough
        sf = (1.0 if a.ssd is not None else 0.5) if a.shallow_frac is None else a.shallow_frac
        D, t0, T, vgrid = sst.stochastic_scenario(
            s_src, z_src, L, W, hs, a.hypo_depth, Dmean=a.slip, Dmax=a.max_slip, vr=a.vr, vr_std=a.vr_std, vr_seg=a.vr_seg,
            vr_seg_z=a.vr_seg_z, vr_min=a.vr_min, vr_max=a.vr_max, vr_step=a.vr_step, rise=a.rise, taper_s=a.taper_s, taper_z=a.taper_z,
            slip_sigma=sig, corr_s=a.corr_s, corr_z=a.corr_z, ssd_pct=a.ssd, ssd_depth=a.ssd_depth, ssd_smooth=a.ssd_smooth,
            shallow_frac=sf, seed=a.seed)
        print(f'stochastic source: seed {a.seed}, slip sigma {sig:g}, rupture speed {a.vr:g} +- {a.vr_std:g} m/s'
              + (f', shallow deficit {a.ssd:g}% above {a.ssd_depth:g} m (smooth {a.ssd_smooth:g})' if a.ssd is not None else ''))
        if a.ssd is not None:
            below = z_src >= a.ssd_depth
            print(f'realized: slip in the top cell row = {100*np.mean(D[:, 0]) / np.mean(D[:, below]):.0f}% of the mean slip below {a.ssd_depth:g} m '
                  f'(target {a.ssd:g}%)')
    M0tot = (med[4] * h * (W / nz) * D).sum()
    print(f'mean slip {D.mean():.2f} m, max {D.max():.2f} m, M0 = {M0tot:.3e} N m  (Mw {2/3*(np.log10(M0tot)-9.1):.2f}), '
          f'rupture duration {t0[D > 0.05 * D.max()].max():.1f} s')

    rise_ref = a.rise
    if a.rise_mode == 'gp':
        T, ri = sst.gp_rise_time(D, z_src, M0tot, mean=a.rise_mean, eta=a.rise_eta, depth_factor=not a.no_rise_depth_factor, corr_s=a.corr_s, corr_z=a.corr_z,
                                 ds=L / ns, dz=W / nz, seed=a.seed + 7919)
        rise_ref = ri['p10']
        print(f"rise time (Graves & Pitarka 2016): fault average {ri['mean']:.2f} s ({'pinned' if a.rise_mean else 'self-similar, 1.6e-9 M0^(1/3)'}), 10-90% {ri['p10']:.2f}-{ri['p90']:.2f} s, "
              f"range {ri['tmin']:.2f}-{ri['tmax']:.2f} s, corr(sqrt slip, rise) = {ri['corr_slip']:.2f}, eta {a.rise_eta:g}, shallow doubling {'off' if a.no_rise_depth_factor else 'on'}")
    else:
        rise_ref = float(np.median(T[D > 0.1 * D.max()])) if np.ndim(T) else float(T)
    fine, coarse = build_sources(trace, s_src, z_src, D, t0, T, med[4], a.blocks, a.blocks)

    offs = np.array([float(v) for v in a.offsets.split(',')])
    offs = np.r_[-offs[::-1], offs]
    z_rec = np.arange(0.5 * a.dz, W, a.dz)
    if a.transects:
        cen, nrm, s_rec = read_transects(a.transects, trace)
        tsel = select_transects(a.transect_subset, len(cen)); cen, nrm, s_rec = cen[tsel], nrm[tsel], s_rec[tsel]
        xy = cen[:, None, None, :] + offs[None, :, None, None] * nrm[:, None, None, :]
        xy = np.broadcast_to(xy, (len(cen), len(offs), len(z_rec), 2))
        zz = np.broadcast_to(-z_rec[None, None, :], xy.shape[:3])
        recg = np.concatenate([xy, zz[..., None]], -1)
        rec, shape = recg.reshape(-1, 3), recg.shape[:3]
    else:
        s_rec = np.arange(0.5 * a.ds, L, a.ds)
        tsel = select_transects(a.transect_subset, len(s_rec)); s_rec = s_rec[tsel]
        rec, shape = build_receivers(trace, s_rec, offs, z_rec)
    if a.transect_subset not in ('', 'all'):
        print(f'transect subset {a.transect_subset!r}: {len(tsel)} transects used (numbers in the figures count 1..{len(tsel)}; numbers in the full set: ' + ','.join(str(int(i) + 1) for i in tsel) + ')')
    depth = -rec[:, 2]
    reset_stress_params(); set_stress_params(a.litho_coef, a.clamp_depth, a.shear_ratio, a.layers)
    if a.seissol_yaml:
        use_seissol_yaml(a.seissol_yaml)
        print(f'initial stress from {a.seissol_yaml} (formula settings and --shmax ignored)')
    else:
        print(f'initial stress from the depth formula: mean = {a.litho_coef:g} g int(rho dz) down to {a.clamp_depth:g} m, deviatoric/mean = {a.shear_ratio:.4f}, SHmax azimuth {a.shmax:g}')
    init = initial_stress(depth, a.shmax)
    coh = np.full(len(rec), a.coh)
    phi = np.full(len(rec), np.arctan(a.bulk_friction))
    pf = np.zeros(len(rec))
    nt = int(a.tmax / a.dt) + 1
    if a.dt > rise_ref / 3.0 + 1e-9:
        print(f'WARNING: time step DT = {a.dt:g} s is coarse for a rise time of {rise_ref:g} s.  The stress peak that moves with the rupture front lasts about '
              f'one rise time, so peak F (and the yielded area) is UNDERESTIMATED when DT > rise/3 (= {rise_ref/3:.2f} s).  Use DT <= {rise_ref/6:.2f} s for converged peaks '
              f'(quasi-static solvers: the extra time steps are almost free).')
    if a.r_near is None:
        a.r_near = 3.0 * a.blocks * h
    omin_ = np.abs(offs).min()
    if a.solver == 'ar' and omin_ < 3.0 * h:
        print(f'WARNING: smallest offset {omin_:.0f} m < 3h = {3*h:.0f} m. The point-source (ar) solver is only about 3% accurate '
              f'from 3h outward (about 25% at 1.5h). Use --solver qs-full / qs-half, or a smaller --h.')
    print(f'{len(rec)} receivers ({shape}), {nt} time steps; running ...', flush=True)
    t1 = time.time()
    t_sl = np.array([float(v) for v in a.save_times.split(',') if v.strip()])
    kidx = np.clip(np.round(t_sl / a.dt).astype(np.int64), 0, nt - 1)
    nsus = max(1, int(np.ceil(a.tv / a.dt - 1e-9))) if a.tv > 0 else 1
    if a.solver == 'ar':
        peakF, tpk, finF, peakSJ, dsm_min, F0, Fsl, dfin, dsl, Fmx, Fsus, Fsus_sl = ks.yield_screen(rec, init, coh, phi, pf, fine, coarse, med,
                                                                           a.dt, nt, a.r_near, a.rcut, a.eps_frac * h, kidx, nsus)
    else:
        import qs_stress as qs                            # needs: pip install cutde
        nu = med[3] / (2 * (med[3] + med[4]))
        cache = None if a.no_kernel_cache else (a.kernel_cache_dir or os.path.join(os.path.dirname(os.path.abspath(a.out)), '_kernel_cache'))
        peakF, tpk, finF, peakSJ, dsm_min, F0, Fsl, dfin, dsl, Fmx, Fsus, Fsus_sl = qs.qs_screen(
            rec.reshape(shape + (3,)), s_rec, z_rec, trace, s_src, z_src, D, t0, T, np.arange(nt) * a.dt, med[4], nu,
            init, coh, phi, pf, a.dt, kidx, mode='full' if a.solver == 'qs-full' else 'half', theta=a.qs_theta,
            cache_dir=cache, cache_max_gb=a.kernel_cache_max_gb, nsus=nsus)
    print(f'solver {time.time()-t1:.0f} s; yield potential (peak F>0) at {100*(peakF>0).mean():.1f}% of receivers, '
          f'final F>0 at {100*(finF>0).mean():.1f}%, max peak F = {peakF.max()/1e6:.2f} MPa')
    print(f'sustained yield (F>0 at {nsus} consecutive time step(s); Tv = {a.tv:g} s, DT = {a.dt:g} s): {100*(Fsus>0).mean():.1f}% of receivers' + ('   [DT >= Tv: cannot resolve Tv, same as peak F>0]' if nsus == 1 and a.tv > 0 else ''))

    # --- on-fault consistency diagnostic: final shear-stress change resolved on the fault vs initial shear stress
    io = [int(np.argmin(np.abs(offs - o))) for o in (-offs[offs > 0].min(), offs[offs > 0].min())]
    sel = rec.reshape(shape + (3,))[:, io, :, :].reshape(-1, 3)
    Sh = dfin.reshape(shape + (6,))[:, io, :, :].reshape(-1, 6)
    tg, nm = trace.tang(s_rec), trace.normal(s_rec)
    TG = np.repeat(tg[:, None, :], len(io) * len(z_rec), 1).reshape(-1, 2)
    NM = np.repeat(nm[:, None, :], len(io) * len(z_rec), 1).reshape(-1, 2)
    dtau = TG[:, 0] * NM[:, 0] * Sh[:, 0] + TG[:, 1] * NM[:, 1] * Sh[:, 1] + (TG[:, 0] * NM[:, 1] + TG[:, 1] * NM[:, 0]) * Sh[:, 3]
    i0 = initial_stress(-sel[:, 2], a.shmax)
    tau0 = TG[:, 0] * NM[:, 0] * i0[:, 0] + TG[:, 1] * NM[:, 1] * i0[:, 1] + (TG[:, 0] * NM[:, 1] + TG[:, 1] * NM[:, 0]) * i0[:, 3]
    sn0 = -(NM[:, 0] ** 2 * i0[:, 0] + NM[:, 1] ** 2 * i0[:, 1] + 2 * NM[:, 0] * NM[:, 1] * i0[:, 3])
    dtau, tau0, sn0 = (v.reshape(len(s_rec), len(io), len(z_rec)) for v in (dtau, tau0, sn0))
    wgt = D[:, :].mean() > 0
    mid = (z_rec > 2000) & (z_rec < 12000)
    print(f'on-fault check (offset +-{offs[offs>0].min():.0f} m, 2-12 km depth): initial tau_tn = {tau0[:, :, mid].mean()/1e6:.1f} MPa '
          f'(must be >0 for right-lateral), mean static change = {dtau[:, :, mid].mean()/1e6:.1f} MPa  '
          f'(10-90%: {np.percentile(dtau[:, :, mid], 10)/1e6:.1f} .. {np.percentile(dtau[:, :, mid], 90)/1e6:.1f})')
    shp = lambda v: v.reshape(shape)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out + '.npz', transect_index=tsel + 1, s_rec=s_rec, offsets=offs, depth_rec=z_rec, rec=rec.reshape(shape + (3,)),
                        peakF=shp(peakF), tpeak=shp(tpk), finalF=shp(finF), F_slices=Fsl.reshape(shape + (len(kidx),)), Fmax_slices=Fmx.reshape(shape + (len(kidx),)), sustF=shp(Fsus), sustF_slices=Fsus_sl.reshape(shape + (len(kidx),)), tv=a.tv, nsus=nsus, dsigma_slices=dsl.reshape(shape + (len(kidx), 6)), t_slices=kidx * a.dt, peakSJ2=shp(peakSJ),
                        dmeanstress_min=shp(dsm_min), F0=shp(F0), L=L, W=W, dsigma_final=dfin.reshape(shape + (6,)),
                        dtau_fault=dtau, tau0_fault=tau0, sn0_fault=sn0, s_src=s_src, z_src=z_src, slip=D, rt=t0, rise=T, vr=(vgrid if vgrid is not None else np.zeros((1, 1))), hypo_s=hs, hypo_depth=a.hypo_depth,
                        params=json.dumps(vars(a)), trace=trace.pos(np.linspace(0, L, 400)),
                        field_raw=field_raw if field_raw is not None else np.zeros((0, 3)))
    write_vtk(a.out + '.vtk', rec.reshape(shape + (3,)), {'peakF_MPa': shp(peakF) / 1e6, 'finalF_MPa': shp(finF) / 1e6,
                                                           'peakF_pos_MPa': np.maximum(shp(peakF), 0) / 1e6,
                                                           'tpeak_s': shp(tpk), 'dMeanStressMin_MPa': shp(dsm_min) / 1e6})
    print(f'wrote {a.out}.npz / .vtk   (total {time.time()-t_start:.0f} s)')


def write_vtk(path, pts, fields):
    """Legacy ASCII structured grid; dims = (n_s, n_offset, n_depth)."""
    ns, no, nz = pts.shape[:3]
    # VTK wants x fastest: reorder (s,o,z) -> index i + ns*(j + no*k)
    P = np.transpose(pts, (2, 1, 0, 3)).reshape(-1, 3)
    with open(path, 'w') as f:
        f.write('# vtk DataFile Version 3.0\nkinematic yield screening\nASCII\nDATASET STRUCTURED_GRID\n')
        f.write(f'DIMENSIONS {ns} {no} {nz}\nPOINTS {len(P)} float\n')
        np.savetxt(f, P, fmt='%.2f')
        f.write(f'POINT_DATA {len(P)}\n')
        for k, v in fields.items():
            f.write(f'SCALARS {k} float 1\nLOOKUP_TABLE default\n')
            np.savetxt(f, np.transpose(v, (2, 1, 0)).reshape(-1), fmt='%.5g')


if __name__ == '__main__':
    main()
