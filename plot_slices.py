#!/usr/bin/env python3
"""
plot_slices.py - time-slice figures from an elsinore_yield.py run.

The run must have been made with save times (SAVE_TIMES in run_spyder.py, or  --save-times 4,8,12  on the command line).
Requested times snap to the nearest time step (dt).  For EVERY saved time t the folder  <run>_slices/tNN.NNs/  gets

    yield_T###.png         yielding cross-section at time t: the MAXIMUM of the yield function F (Drucker-Prager, MPa) reached at any time up to t,
                           i.e. everything that has yielded so far (black line F = 0).  Yielding does not reverse, so this is the map to use.
    yield_now_T###.png     (only with --yield-mode instant or both) F at the instant t itself; it can fall back below 0 after the rupture front passes
    normal_pct_T###.png    change of the stress normal to the fault, in % of the initial normal stress
                           (+ = more compressive / clamping, - = unclamping); black line = F = 0 at time t
    mean_pct_T###.png      change of the mean stress (same % convention);  only with  --stress normal,mean
    j2_pct_T###.png        change of sqrt(J2) (deviatoric stress intensity) in % of the initial sqrt(J2), + = more deviatoric stress;  only with  --stress j2  (or normal,mean,j2)
    fault_slip_rt.png      the fault plane in plan view (along strike x depth): slip accumulated by time t (color),
                           rupture-time contours, the rupture front at time t in red, the hypocenter as a star

One PNG per transect; use --transects 10,31,62 to limit the number (default: all).

  python plot_slices.py results/example.npz
  python plot_slices.py results/example.npz --transects 31,62,95 --stress normal --dpi 110
"""
import argparse, os, json
import numpy as np, matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
import plot_sections as ps

INK, MUTE = ps.INK, ps.MUTE


def initial_tensor(d):
    """Initial stress (Pa, compression negative; order xx yy zz xy yz xz) at every receiver, as used in the run."""
    import elsinore_yield as ey
    p = json.loads(str(d['params']))
    ey.restore_stress_from_params(p)
    zz = np.broadcast_to(np.asarray(d['depth_rec'], float)[None, None, :], np.asarray(d['rec']).shape[:3])
    return ey.initial_stress(zz, float(p.get('shmax', 350.0)))


def fault_normals(d):
    """Horizontal fault-normal unit vector (x, y) of each transect, from the receiver positions (offsets are measured along it)."""
    rec = np.asarray(d['rec'])
    v = rec[:, -1, 0, :2] - rec[:, 0, 0, :2]
    return v / np.hypot(v[:, 0], v[:, 1])[:, None]


def normal_component(n, T):
    """n^T T n for tensors T[..., 6] (xx yy zz xy yz xz) and normals n[nsr, 2] that vary along axis 0 of T."""
    nx = n[:, 0][:, None, None]; ny = n[:, 1][:, None, None]
    return nx * nx * T[..., 0] + ny * ny * T[..., 1] + 2 * nx * ny * T[..., 3]


def slip_at(d, t):
    """Slip accumulated at time t (raised-cosine slip rate with the saved rupture time and rise time)."""
    x = np.clip((t - np.asarray(d['rt'])) / np.maximum(np.asarray(d['rise']), 1e-9), 0.0, 1.0)
    return np.asarray(d['slip']) * 0.5 * (1.0 - np.cos(np.pi * x))


def plot_fault_slice(d, t, out, contour_dt=1.0, slip_max=None, dpi=150):
    """Plan view of the fault plane at time t: slip so far (color) + rupture-time contours + current front."""
    s = np.asarray(d['s_src']) / 1e3; z = np.asarray(d['z_src']) / 1e3
    rt = np.asarray(d['rt']); D = np.asarray(d['slip']); W = float(d['W']) / 1e3
    vmax = slip_max or float(D.max())
    fig, ax = plt.subplots(figsize=(11, 4.0))
    im = ax.pcolormesh(s, z, slip_at(d, t).T, cmap='YlOrBr', shading='nearest', vmin=0, vmax=vmax)
    lev = np.arange(0, np.ceil(rt[D > 0.05 * D.max()].max()) + contour_dt, contour_dt)
    cs = ax.contour(s, z, rt.T, levels=lev, colors=INK, linewidths=0.7)
    ax.clabel(cs, fmt='%g s', fontsize=7, inline=True)
    if rt.min() < t < rt.max():
        ax.contour(s, z, rt.T, levels=[t], colors='#c0392b', linewidths=2.2)     # rupture front at time t
    ax.plot(float(d['hypo_s']) / 1e3, float(d['hypo_depth']) / 1e3, '*', color='white', mec=INK, ms=14, mew=1)
    ax.set_ylim(W, 0); ax.set_xlim(float(d['L']) / 1e3, 0)
    ax.set_xlabel('along-strike distance from SE end (km)'); ax.set_ylabel('depth (km)')
    ax.text(0.0, 1.11, 'NW', transform=ax.transAxes, color=MUTE, fontweight='bold')
    ax.text(1.0, 1.11, 'SE', transform=ax.transAxes, ha='right', color=MUTE, fontweight='bold')
    fig.colorbar(im, ax=ax, pad=0.01).set_label(f'slip at t = {t:g} s (m)')
    ax.set_title(f'Fault plane, t = {t:g} s: slip so far, rupture-time contours every {contour_dt:g} s, red = rupture front, star = hypocenter',
                 loc='left', fontsize=9, pad=34)
    fig.savefig(out, dpi=dpi, bbox_inches='tight'); plt.close(fig)


def make_slices(npz, outdir=None, transects='all', stress=('normal', 'mean'), dpi=130, pct_max=None, yield_max=None,
                contour_dt=1.0, slip_max=None, verbose=True, yield_mode='max', levels=(0.0,), guides=True):
    d = np.load(npz); base = npz.replace('.npz', '')
    outdir = outdir or base + '_slices'
    if 't_slices' not in d.files or len(d['t_slices']) == 0:
        raise SystemExit('this run has no time slices: set SAVE_TIMES = [4, 8, 12] (or --save-times 4,8,12) and rerun')
    ts_all = np.asarray(d['t_slices']); _, first = np.unique(ts_all, return_index=True); order = np.sort(first)
    off = np.asarray(d['offsets']); z = np.asarray(d['depth_rec']) / 1e3; s = np.asarray(d['s_rec']) / 1e3
    omin = np.abs(off).min(); oo = np.arange(off.min(), off.max() + 1, 10.0)
    ids = list(range(len(s))) if transects == 'all' else [int(v) - 1 for v in str(transects).split(',')]
    zz = ps.smooth_grid(off, z, np.zeros((1, len(off), len(z))), oo)[1]            # finer depth grid for display
    interp = lambda A: ps.smooth_grid(off, z, A, oo)[0]
    Fnow = np.asarray(d['F_slices']) / 1e6                               # (nsr, no, nz, nk): F at the instant
    if 'Fmax_slices' in d.files:
        Fmx = np.asarray(d['Fmax_slices']) / 1e6                         # running maximum up to each slice time
    else:
        Fmx = Fnow
        if yield_mode != 'instant':
            print('NOTE: this run was made before the running maximum was stored; the yield figures show F at the instant (rerun to get the maximum so far)')
    levels = [float(v) for v in levels]
    ps.GUIDES = bool(guides)
    SUS = None; tvv = None                                                          # F > 0 sustained for >= Tv, running maximum up to each slice time
    if 'sustF_slices' in d.files and float(d['tv']) > 0 and int(d['nsus']) > 1 and yield_mode in ('max', 'both'):
        SUS = np.asarray(d['sustF_slices']) / 1e6; tvv = float(d['tv'])
    F = Fmx if yield_mode in ('max', 'both') else Fnow                  # field for the yield figures and the F = 0 outline on the stress figures
    posF = F[F > 0]
    yv = yield_max or max(np.percentile(posF, 98) if posF.size else 1.0, 1.0)
    fields = {}
    if stress:
        dsl = np.asarray(d['dsigma_slices'])                             # (nsr, no, nz, nk, 6)
        i0 = initial_tensor(d); n = fault_normals(d)
        sn0 = normal_component(n, i0); sm0 = i0[..., :3].mean(-1); sj0 = ps.sqrt_j2(i0)
        pct = {}
        for j in order:
            if 'normal' in stress: pct[('normal', j)] = 100.0 * normal_component(n, dsl[..., j, :]) / sn0
            if 'mean' in stress: pct[('mean', j)] = 100.0 * dsl[..., j, :3].mean(-1) / sm0
            if 'j2' in stress: pct[('j2', j)] = 100.0 * (ps.sqrt_j2(i0 + dsl[..., j, :]) - sj0) / np.maximum(sj0, 1.0)
        fields = pct
    pv = {}
    for k in ('normal', 'mean', 'j2'):
        v = [np.abs(fields[(k, j)]) for j in order if (k, j) in fields]
        if v: pv[k] = pct_max or max(float(np.percentile(np.concatenate([x.ravel() for x in v]), 98)), 1e-6)
    lab = {'normal': ('normal_pct', 'fault-normal stress change (% of initial); + = clamping',
                      'normal-stress change'),
           'mean': ('mean_pct', 'mean-stress change (% of initial); + = more compressive', 'mean-stress change'),
           'j2': ('j2_pct', 'sqrt(J2) change (% of initial); + = more deviatoric stress', 'sqrt(J2) change')}
    nfig = 0
    for j in order:
        t = float(ts_all[j]); sub = os.path.join(outdir, f't{t:05.2f}s'); os.makedirs(sub, exist_ok=True)
        plot_fault_slice(d, t, os.path.join(sub, 'fault_slip_rt.png'), contour_dt, slip_max, dpi=dpi); nfig += 1
        Fr = interp(F[..., j]); label = f't = {t:g} s'
        tvF = None if SUS is None else interp(SUS[..., j])
        ylab = f't = {t:g} s, maximum so far' if yield_mode in ('max', 'both') else f't = {t:g} s, at this instant'
        for i in ids:
            ps.single_section(d, i, oo, zz, Fr, s, Normalize(0, yv), omin, ylab, os.path.join(sub, f'yield_T{i+1:03d}.png'),
                              cblabel='F (MPa);  F>0 outlined' if len(levels) == 1 else 'F (MPa);  outlines at the levels in the legend', dpi=dpi, levels=levels, tvF=tvF if yield_mode != 'instant' else None, tv=tvv if SUS is not None else None); nfig += 1
        if yield_mode == 'both':
            Fi = interp(Fnow[..., j])
            for i in ids:
                ps.single_section(d, i, oo, zz, Fi, s, Normalize(0, yv), omin, f't = {t:g} s, at this instant', os.path.join(sub, f'yield_now_T{i+1:03d}.png'),
                                  cblabel='F (MPa);  F>0 outlined' if len(levels) == 1 else 'F (MPa);  outlines at the levels in the legend', dpi=dpi, levels=levels, tvF=None, tv=None); nfig += 1
        for kind in [k for k in ('normal', 'mean', 'j2') if (k, j) in fields]:
            A = interp(fields[(kind, j)]); fn, cbl, _ = lab[kind]
            for i in ids:
                ps.single_section(d, i, oo, zz, A, s, Normalize(-pv[kind], pv[kind]), omin, label + (' (outlines: F, maximum so far)' if yield_mode in ('max', 'both') else ' (outlines: F at this instant)'), os.path.join(sub, f'{fn}_T{i+1:03d}.png'),
                                  cblabel=cbl, cmap='RdBu_r', clip=False, overlay=Fr, dpi=dpi, levels=levels, tvF=tvF, tv=tvv if SUS is not None else None); nfig += 1
        if verbose: print(f'   slice t = {t:g} s done ({nfig} figures so far) -> {sub}/', flush=True)
    return outdir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz'); ap.add_argument('--out', default=None, help='output folder (default <run>_slices)')
    ap.add_argument('--transects', default='all'); ap.add_argument('--stress', default='normal,mean', help="any of 'normal', 'mean', 'j2' (comma separated) or 'none'")
    ap.add_argument('--dpi', type=int, default=130); ap.add_argument('--pct-max', type=float, default=None)
    ap.add_argument('--yield-max', type=float, default=None); ap.add_argument('--contour-dt', type=float, default=1.0)
    ap.add_argument('--slip-max', type=float, default=None)
    ap.add_argument('--no-guides', action='store_true', help='no dashed 1.5h / 3h lines')
    ap.add_argument('--levels', default='0', help='F levels (MPa) to outline, e.g. 0,5,20')
    ap.add_argument('--yield-mode', default='max', choices=['max', 'instant', 'both'], help="yield figures: 'max' = maximum so far (default), 'instant' = F at time t, 'both'")
    a = ap.parse_args(argv)
    st = () if a.stress == 'none' else tuple(v.strip() for v in a.stress.split(','))
    o = make_slices(a.npz, a.out, a.transects, st, a.dpi, a.pct_max, a.yield_max, a.contour_dt, a.slip_max, yield_mode=a.yield_mode, levels=[float(v) for v in a.levels.split(',')], guides=not a.no_guides)
    print('wrote slice figures in', o)


if __name__ == '__main__':
    matplotlib.use('Agg')
    main()
