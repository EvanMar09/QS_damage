#!/usr/bin/env python3
"""
plot_sections.py - figures from an elsinore_yield.py run.

  python plot_sections.py run.npz                      # slip + rupture-time figure, and one PNG per cross-section in <run>_sections_final/
  python plot_sections.py run.npz --sheet              # also the big contact sheet
  python plot_sections.py run.npz --what peak          # peak-over-time instead of final
  python plot_sections.py run.npz --what dmean         # final mean-stress change (MPa) with the F>0 outline;  --what dmeanpct = % of initial
  python plot_sections.py run.npz --what dj2pct        # final change of sqrt(J2) in % of the initial sqrt(J2) (--what dj2 = MPa)
  python plot_sections.py run.npz --what time --time 7 # time slice (run must have been made with --save-times 7,...)
  python plot_sections.py run.npz --transects 1,12,30 --pdf    # subset of sections (default: all), plus a multi-page PDF
"""
import argparse, numpy as np, matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d, PchipInterpolator
import json
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

INK, MUTE = '#222222', '#777777'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTE, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def plot_slip_rt(d, out, contour_dt=1.0, depth_max=None, slip_max=None):
    """Slip (color) with rupture-time contours; along-strike distance from the SE end on x, depth on y
    (0 km at the top, fault base at the bottom)."""
    s = d['s_src'] / 1e3; z = d['z_src'] / 1e3
    D = d['slip']; rt = d['rt']
    W = float(d['W']) / 1e3 if depth_max is None else depth_max
    fig, ax = plt.subplots(figsize=(11, 4.0))
    im = ax.pcolormesh(s, z, D.T, cmap='YlOrBr', shading='nearest', vmin=0, vmax=slip_max)
    levels = np.arange(0, np.ceil(rt[D > 0.05 * D.max()].max()) + contour_dt, contour_dt)
    cs = ax.contour(s, z, rt.T, levels=levels, colors=INK, linewidths=0.8)
    ax.clabel(cs, fmt='%g s', fontsize=7, inline=True)
    ax.plot(float(d['hypo_s']) / 1e3, float(d['hypo_depth']) / 1e3, '*', color='white', mec=INK, ms=14, mew=1)
    ax.set_ylim(W, 0)                                    # 0 at top, fault base at bottom
    ax.set_xlim(float(d['L']) / 1e3, 0)                  # SE end on the RIGHT
    ax.set_xlabel('along-strike distance from SE end (km)'); ax.set_ylabel('depth (km)')
    ax.text(0.0, 1.11, 'NW', transform=ax.transAxes, color=MUTE, fontweight='bold'); ax.text(1.0, 1.11, 'SE', transform=ax.transAxes, ha='right', color=MUTE, fontweight='bold')
    if 's_rec' in d.files:                               # one dot per cross-section on the surface
        sr = d['s_rec'] / 1e3
        ax.plot(sr, np.zeros_like(sr), 'o', ms=2.2, color=INK, clip_on=False, zorder=6)
        step = max(1, int(round(len(sr) / 12 / 5)) * 5)
        for i in range(step - 1, len(sr), step):
            ax.annotate(f'T{i+1}', (sr[i], 0), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=6, color=MUTE, annotation_clip=False)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label('final slip (m)')
    ax.set_title(f'Slip with rupture-time contours every {contour_dt:g} s   (star = hypocenter)', loc='left', fontsize=9.5, pad=34)
    fig.savefig(out, dpi=170, bbox_inches='tight'); plt.close(fig)


def plot_depth_profile(d, out, depth_max=None):
    """Slip against depth: along-strike mean (line) and 10-90 % range (band), with the shallow transition layer marked."""
    D = np.asarray(d['slip']); z = np.asarray(d['z_src']) / 1e3
    try:
        p = json.loads(str(d['params']))
    except Exception:
        p = {}
    layer = None
    if p.get('field_slip') and p.get('depth_scale', 1.0) != 1.0:
        layer = (p['depth_scale_depth'], p.get('depth_scale_smooth', 0.0), f"x{p['depth_scale']:g} below")
    elif p.get('ssd') is not None:
        layer = (p['ssd_depth'], p.get('ssd_smooth', 0.0), f"{p['ssd']:g}% at surface")
    fig, a = plt.subplots(figsize=(4.6, 5.0))
    a.fill_betweenx(z, np.percentile(D, 10, axis=0), np.percentile(D, 90, axis=0), color='#f4b183', alpha=0.5, lw=0, label='10-90 % along strike')
    a.plot(D.mean(axis=0), z, color='#c0392b', lw=1.6, label='along-strike mean')
    if layer:
        zt, sm, lab = layer; z1 = zt * (1 - sm)
        a.axhspan(0, zt / 1e3, color='#dddddd', alpha=0.5, zorder=0)
        a.axhline(zt / 1e3, color='k', lw=0.8, ls=(0, (4, 3)))
        if sm > 0: a.axhline(z1 / 1e3, color='#1f5fd6', lw=0.8, ls=(0, (4, 3)))
        a.set_title(f'shallow layer {zt/1e3:g} km ({lab}), smoothing = {sm:g}', fontsize=9, loc='left')
    a.set_ylim((depth_max or z.max()), 0); a.set_xlim(left=0)
    a.set_xlabel('slip (m)'); a.set_ylabel('depth (km)'); a.legend(fontsize=7, loc='lower right', frameon=False)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)


def mean_stress_change(d):
    """Final mean-stress change (Pa, compression negative) from the saved tensor, and the initial mean stress (Pa) at each receiver."""
    dm = np.asarray(d['dsigma_final'])[..., :3].mean(axis=-1)
    try: shmax = float(json.loads(str(d['params'])).get('shmax', 350.0))
    except Exception: shmax = 350.0
    import elsinore_yield as ey
    try: ey.restore_stress_from_params(json.loads(str(d['params'])))
    except Exception: pass
    zz = np.broadcast_to(np.asarray(d['depth_rec'])[None, None, :], dm.shape)
    i0 = ey.initial_stress(zz, shmax)
    return dm, i0[..., :3].mean(axis=-1)


def sqrt_j2(T):
    """sqrt(J2) of the deviatoric stress for tensors in Voigt-like order (xx, yy, zz, xy, yz, xz) on the last axis (any units)."""
    T = np.asarray(T, float)
    m = T[..., :3].mean(-1, keepdims=True)
    dxx, dyy, dzz = T[..., 0] - m[..., 0], T[..., 1] - m[..., 0], T[..., 2] - m[..., 0]
    return np.sqrt(np.maximum(0.5 * (dxx ** 2 + dyy ** 2 + dzz ** 2) + T[..., 3] ** 2 + T[..., 4] ** 2 + T[..., 5] ** 2, 0.0))


def initial_tensor_rec(d):
    """Initial stress tensor (Pa) at every receiver, as used in the run."""
    try: p = json.loads(str(d['params']))
    except Exception: p = {}
    import elsinore_yield as ey
    try: ey.restore_stress_from_params(p)
    except Exception: pass
    zz = np.broadcast_to(np.asarray(d['depth_rec'], float)[None, None, :], np.asarray(d['dsigma_final'])[..., 0].shape)
    return ey.initial_stress(zz, float(p.get('shmax', 350.0)))


def j2_change(d):
    """Final change of sqrt(J2) (Pa; + = more deviatoric stress) and the initial sqrt(J2) (Pa) at each receiver."""
    i0 = initial_tensor_rec(d)
    s0 = sqrt_j2(i0)
    return sqrt_j2(i0 + np.asarray(d['dsigma_final'])) - s0, s0


def get_field(d, what, time=None):
    if what in ('dmean', 'dmeanpct'):
        dm, sm0 = mean_stress_change(d)
        if what == 'dmean':
            return dm / 1e6, 'final mean-stress change'
        return 100.0 * dm / sm0, 'final mean-stress change, % of initial'
    if what in ('dj2', 'dj2pct'):
        dj, sj0 = j2_change(d)
        if what == 'dj2':
            return dj / 1e6, 'final sqrt(J2) change'
        return 100.0 * dj / np.maximum(sj0, 1.0), 'final sqrt(J2) change, % of initial'
    if what == 'final':
        return d['finalF'] / 1e6, 'final state'
    if what == 'peak':
        return d['peakF'] / 1e6, 'peak over time'
    if what == 'sustained':
        return d['sustF'] / 1e6, f'peak over time, sustained >= Tv = {float(d["tv"]):g} s'
    ts = d['t_slices']
    if len(ts) == 0:
        raise SystemExit('this run has no time slices: rerun elsinore_yield.py with --save-times 5,8,12')
    j = int(np.argmin(abs(ts - time)))
    return d['F_slices'][..., j] / 1e6, f't = {ts[j]:g} s'


GUIDES = True     # False: no 1.5h / 3h guide lines and no dashed outlines inside 1.5h (set by --no-guides / the GUIDE_LINES setting)


def get_h(d):
    """Patch / receiver cell size h (m) stored with a run (or suite summary); None if unknown or if the guides are switched off."""
    if not GUIDES: return None
    try:
        if 'h' in d.files: return float(d['h'])
        return float(json.loads(str(d['params']))['h'])
    except Exception:
        return None


def _clip(cs, patch):
    try: cs.set_clip_path(patch)
    except Exception:
        for c in cs.collections: c.set_clip_path(patch)


LEVEL_COLORS = ['#111111', '#6a1b9a', '#00796b', '#e65100', '#5d4037', '#455a64']     # outline color for each F level, in the order given
TV_COLOR = '#d81b60'


def outline(a, oo, z, A, level=0.0, h=None, color=INK, lw=0.7):
    """Black F = level outline.  Receivers closer to the fault than 1.5h are the least accurate (the patch solver), so
    inside +-1.5h the outline is dashed; outside it is solid."""
    if not h or abs(1.5 * h) >= max(abs(oo.min()), abs(oo.max())) or A.min() > level or A.max() < level:
        if A.min() <= level <= A.max(): a.contour(oo, z, A, levels=[level], colors=color, linewidths=lw)
        return
    x0, x1, y0, y1 = oo.min() - 1, oo.max() + 1, -1.0, z.max() + 1.0
    for (xa, xb, ls) in ((x0, -1.5 * h, '-'), (1.5 * h, x1, '-'), (-1.5 * h, 1.5 * h, '--')):
        cs = a.contour(oo, z, A, levels=[level], colors=color, linewidths=lw, linestyles=ls)
        _clip(cs, Rectangle((xa, y0), xb - xa, y1 - y0, transform=a.transData))


def smooth_grid(off, z, F, oo, zmax_step=0.05):
    """Display grid: shape-preserving (PCHIP, no overshoot) interpolation of F (ntr, noff, nz) along offset onto oo and along
    depth onto a grid of at most 50 m spacing.  Linear interpolation across the coarse receiver spacing (hundreds of m)
    draws long straight kinked outlines that are not in the data.  Returns (Fr, zz)."""
    Fr = PchipInterpolator(off, F, axis=1)(oo)
    if len(z) > 1 and np.diff(z).max() > zmax_step * 1.2:
        zz = np.linspace(z.min(), z.max(), int(np.ceil((z.max() - z.min()) / zmax_step)) + 1)
        Fr = PchipInterpolator(z, Fr, axis=2)(zz)
    else:
        zz = z
    return Fr, zz


def section_axes(a, oo, z, A, norm, omin, level=0.0, h=None, cmap='Reds', clip=True, overlay=None, levels=None, tvF=None, tv=None):
    """Yield-potential section.  Everything between the nearest receivers (+-omin) is unsampled (gray).
    If h is given, dashed lines mark the accuracy guides: 1.5h (black) and 3h (blue) on either side of the fault."""
    im = a.pcolormesh(oo, z, np.maximum(A, 0) if clip else A, cmap=cmap, norm=norm, shading='nearest')
    lev = [level] if levels is None else list(levels)
    Fo = A if overlay is None else overlay                              # overlay: F outlines on a non-F field
    hl = []
    for q, lv in enumerate(lev):                                         # one outline per F level (MPa), different colors
        col = LEVEL_COLORS[q % len(LEVEL_COLORS)]
        outline(a, oo, z, Fo, lv, h, color=col, lw=0.8 if q == 0 else 1.0)
        hl.append(Line2D([], [], color=col, lw=1.2, label=f'F = {lv:g} MPa'))
    if tvF is not None:                                                  # F > 0 sustained for at least Tv
        outline(a, oo, z, tvF, 0.0, h, color=TV_COLOR, lw=1.0)
        hl.append(Line2D([], [], color=TV_COLOR, lw=1.2, label=f'F > 0 for >= Tv = {tv:g} s'))
    if len(hl) == 1: hl = []
    a.axvspan(-omin, omin, color='#dddddd', zorder=3)
    a.axvline(0, color=MUTE, lw=0.5, zorder=4)
    if h:
        for k, col in ((1.5, 'k'), (3.0, '#1f5fd6')):
            for sgn in (-1, 1):
                if abs(k * h) < max(abs(oo.min()), abs(oo.max())):
                    a.axvline(sgn * k * h, color=col, lw=0.9, ls=(0, (4, 3)), zorder=5)
        a.legend(handles=[Line2D([], [], color='k', lw=0.9, ls=(0, (4, 3)), label=f'1.5h = {1.5*h:g} m (outlines dashed inside)'),
                          Line2D([], [], color='#1f5fd6', lw=0.9, ls=(0, (4, 3)), label=f'3h = {3*h:g} m')] + hl,
                 loc='lower center', fontsize=6.0, frameon=True, framealpha=0.88, ncol=3 if hl else 2, borderpad=0.3, handlelength=1.6, columnspacing=1.0)
    elif hl:
        a.legend(handles=hl, loc='lower center', fontsize=6.0, frameon=True, framealpha=0.88, ncol=3, borderpad=0.3, handlelength=1.6)
    a.set_ylim(z.max() + 0.1, 0); a.set_xlim(oo.min(), oo.max())     # 0 km depth at top
    return im


def locator(a, d, i):
    """Small map of the fault trace (rotated so SE is on the right, NE up) with a dot for every cross-section,
    the one drawn here highlighted."""
    tr = np.asarray(d['trace'], float) / 1e3
    v = tr[0] - tr[-1]; v /= np.linalg.norm(v)            # NW -> SE direction
    R = np.array([[v[0], v[1]], [-v[1], v[0]]])
    X = lambda P: (np.atleast_2d(P) - tr[-1]) @ R.T
    T = X(tr)
    ss = d['s_rec']; sT = np.r_[0, np.cumsum(np.hypot(*np.diff(tr, axis=0).T))] * 1e3
    P = np.c_[np.interp(ss, sT, tr[:, 0]), np.interp(ss, sT, tr[:, 1])]
    Q = X(P)
    a.plot(T[:, 0], T[:, 1], '-', color=MUTE, lw=1.2)
    a.plot(Q[:, 0], Q[:, 1], 'o', ms=2.2, color='#999999')
    a.plot(Q[i, 0], Q[i, 1], 'o', ms=6, color='#c0392b', mec='white', mew=0.8, zorder=5)
    a.set_aspect('equal'); a.set_yticks([]); a.set_xticks([])
    for sp in a.spines.values(): sp.set_visible(False)
    a.text(0.0, 0.5, 'NW', transform=a.transAxes, ha='right', va='center', color=MUTE, fontsize=8)
    a.text(1.0, 0.5, 'SE', transform=a.transAxes, ha='left', va='center', color=MUTE, fontsize=8)
    a.text(0.5, 1.0, 'NE', transform=a.transAxes, ha='center', va='bottom', color=MUTE, fontsize=7)


def single_section(d, i, oo, z, Fr, s, norm, omin, label, out=None, pdf=None, cblabel='F (MPa);  F>0 outlined', level=0.0, cmap='Reds', clip=True, overlay=None, dpi=150, levels=None, tvF=None, tv=None):
    fig = plt.figure(figsize=(6.4, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3.6], width_ratios=[40, 1], hspace=0.28, wspace=0.05)
    am = fig.add_subplot(gs[0, 0]); locator(am, d, i)
    a_ = fig.add_subplot(gs[1, 0]); cax = fig.add_subplot(gs[1, 1])
    im = section_axes(a_, oo, z, Fr[i].T, norm, omin, level, h=get_h(d), cmap=cmap, clip=clip,
                     overlay=None if overlay is None else overlay[i].T, levels=levels,
                     tvF=None if tvF is None else tvF[i].T, tv=tv)
    a_.plot(0, 0, 'o', ms=6, color='#c0392b', mec='white', mew=0.8, clip_on=False, zorder=6)   # surface point = this section
    a_.set_xlabel('offset from fault (m)   (- NE | SW +)'); a_.set_ylabel('depth (km)')
    a_.set_title(f'Transect {i+1}, s = {s[i]:.1f} km from SE end, {label}', fontsize=10, loc='left')
    fig.colorbar(im, cax=cax).set_label(cblabel)
    if out: fig.savefig(out, dpi=dpi, bbox_inches='tight')
    if pdf is not None: pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz'); ap.add_argument('--what', default='final', choices=['final', 'peak', 'time', 'dmean', 'dmeanpct', 'dj2', 'dj2pct', 'sustained'])
    ap.add_argument('--time', type=float, default=None); ap.add_argument('--transects', default='all')
    ap.add_argument('--ncols', type=int, default=5); ap.add_argument('--vmax', type=float, default=None)
    ap.add_argument('--contour-dt', type=float, default=1.0); ap.add_argument('--pdf', action='store_true')
    ap.add_argument('--no-slip-fig', action='store_true')
    ap.add_argument('--no-guides', action='store_true', help='do not draw the dashed 1.5h / 3h accuracy lines (outlines are then solid everywhere)')
    ap.add_argument('--levels', default='0', help='F levels (MPa) to outline, e.g. 0,5,20 (default 0)')
    ap.add_argument('--no-tv', action='store_true', help='do not draw the sustained (>= Tv) F>0 outline')
    ap.add_argument('--sheet', action='store_true', help='also make the big contact sheet of all sections')
    a = ap.parse_args(argv)
    d = np.load(a.npz); base = a.npz.replace('.npz', '')
    if not a.no_slip_fig:
        plot_slip_rt(d, base + '_slip_rt.png', a.contour_dt); print('wrote', base + '_slip_rt.png')
    F, label = get_field(d, a.what, a.time)
    tag = a.what if a.what != 'time' else f't{a.time:g}'
    off = d['offsets']; z = d['depth_rec'] / 1e3; s = d['s_rec'] / 1e3
    omin = np.abs(off).min()
    oo = np.arange(off.min(), off.max() + 1, 10.0)
    Fr, zz = smooth_grid(off, z, F, oo)
    ids = list(range(F.shape[0])) if a.transects == 'all' else [int(v) - 1 for v in a.transects.split(',')]
    global GUIDES
    GUIDES = not a.no_guides
    mean_mode = a.what in ('dmean', 'dmeanpct', 'dj2', 'dj2pct')
    levels = [float(v) for v in str(a.levels).split(',') if v.strip() != '']
    tvF = None; tvv = None
    if a.what in ('peak', 'dmean', 'dmeanpct', 'dj2', 'dj2pct') and not a.no_tv and 'sustF' in d.files and float(d['tv']) > 0 and int(d['nsus']) > 1:
        tvF = smooth_grid(off, z, d['sustF'] / 1e6, oo)[0]; tvv = float(d['tv'])     # F > 0 sustained for >= Tv
    if mean_mode:                      # diverging scale centered on 0; the final F > 0 outline is drawn on top
        vmax = a.vmax or max(np.percentile(np.abs(F), 98), 1e-6)
        norm = Normalize(-vmax, vmax); cmap, clip = 'RdBu_r', False
        over = smooth_grid(off, z, d['peakF'] / 1e6, oo)[0]       # outlines: everything that yielded at ANY time (yielding cannot reverse)
        cbl = {'dmean': 'mean-stress change (MPa); negative = more compressive',
               'dmeanpct': 'mean-stress change (% of initial mean stress); + = more compressive',
               'dj2': 'sqrt(J2) change (MPa); + = more deviatoric stress',
               'dj2pct': 'sqrt(J2) change (% of initial sqrt(J2)); + = more deviatoric stress'}[a.what]
    else:
        pos = F[F > 0]
        vmax = a.vmax or max(np.percentile(pos, 98) if pos.size else 1.0, 1.0)
        norm = Normalize(0, vmax); cmap, clip, over = 'Reds', True, None
        cbl = 'F (MPa);  outlines at the levels in the legend' if len(levels) > 1 else 'F (MPa);  F>0 outlined'
    import os
    outdir = f'{base}_sections_{tag}'; os.makedirs(outdir, exist_ok=True)
    for i in ids:                                         # one PNG per cross-section
        single_section(d, i, oo, zz, Fr, s, norm, omin, label, f'{outdir}/T{i+1:03d}.png', cblabel=cbl, cmap=cmap, clip=clip, overlay=over, levels=levels, tvF=tvF, tv=tvv)
    print(f'wrote {len(ids)} PNGs in {outdir}/')
    if a.pdf:
        pdfp = f'{outdir}.pdf'
        with PdfPages(pdfp) as pdf:
            for i in ids:
                single_section(d, i, oo, zz, Fr, s, norm, omin, label, pdf=pdf, cblabel=cbl, cmap=cmap, clip=clip, overlay=over, levels=levels, tvF=tvF, tv=tvv)
        print('wrote', pdfp)
    if a.sheet:
        nc = min(a.ncols, len(ids)); nr = int(np.ceil(len(ids) / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(2.8 * nc, 2.1 * nr), sharex=True, sharey=True, squeeze=False)
        for k, a_ in enumerate(ax.ravel()):
            if k >= len(ids): a_.axis('off'); continue
            i = ids[k]; im = section_axes(a_, oo, zz, Fr[i].T, norm, omin, h=get_h(d), cmap=cmap, clip=clip,
                                         overlay=None if over is None else over[i].T, levels=levels,
                                         tvF=None if tvF is None else tvF[i].T, tv=tvv)
            a_.set_title(f'T{i+1}  s={s[i]:.1f} km', fontsize=7.5, loc='left', color=MUTE, pad=2)
        for a_ in ax[:, 0]: a_.set_ylabel('depth (km)')
        for a_ in ax[-1]: a_.set_xlabel('offset (m, - NE / + SW)')
        fig.subplots_adjust(right=0.9, hspace=0.35)
        cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
        fig.colorbar(im, cax=cax).set_label(f'{cbl}; {label}; gray = no receivers; dashed black/blue = 1.5h / 3h')
        fig.savefig(f'{base}_sections_{tag}.png', dpi=130, bbox_inches='tight'); plt.close(fig)
        print('wrote', f'{base}_sections_{tag}.png')


if __name__ == '__main__':
    matplotlib.use('Agg')
    main()
