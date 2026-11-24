"""
compare_runs.py - compare several runs that share the same slip / rupture-time set-up but use different solvers
(e.g. --solver ar, qs-full, qs-half).  The first run listed is the reference.

  python compare_runs.py ar=run_ar.npz full=run_qf.npz half=run_qh.npz --out cmp
  python compare_runs.py ... --transects 20,62,100 --what final        # per-transect panels (default: 5 spread over the fault)

Writes  <out>_profiles.png   yielding fraction vs depth (final, peak) and stress-change ratio vs the reference
        <out>_scatter.png    stress-change invariant of each run against the reference
        <out>_T###.png       cross-section panels for the chosen transects: F of every run + difference to reference
        <out>_table.txt      numbers
"""
import argparse, numpy as np, matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.interpolate import interp1d
import plot_sections as ps

INK, MUTE = '#222222', '#777777'
STY = [dict(color='#1f4e79', ls='-'), dict(color='#c0392b', ls='--'), dict(color='#2e8b57', ls=':'), dict(color='#8e44ad', ls='-.')]


def sqrtj2(S):
    sm = S[..., :3].mean(-1, keepdims=True)
    d = S[..., :3] - sm
    return np.sqrt(0.5 * (d**2).sum(-1) + S[..., 3]**2 + S[..., 4]**2 + S[..., 5]**2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='label=file.npz  (first = reference)')
    ap.add_argument('--out', default='compare'); ap.add_argument('--what', default='final', choices=['final', 'peak'])
    ap.add_argument('--transects', default=None, help='comma list of transect numbers (1-based)')
    ap.add_argument('--vmax', type=float, default=None)
    a = ap.parse_args(argv)
    R = []
    for r in a.runs:
        lab, f = r.split('=', 1) if '=' in r else (r.replace('.npz', ''), r)
        R.append((lab, np.load(f)))
    ref = R[0][1]
    z = ref['depth_rec'] / 1e3; off = ref['offsets']; s = ref['s_rec'] / 1e3
    for lab, d in R[1:]:
        if d['finalF'].shape != ref['finalF'].shape:
            raise SystemExit(f'{lab}: receiver grid differs from the reference - rerun with the same --ds/--dz/--offsets')
    F = {lab: (d['finalF'] if a.what == 'final' else d['peakF']) / 1e6 for lab, d in R}
    J = {lab: sqrtj2(d['dsigma_final']) / 1e6 for lab, d in R if 'dsigma_final' in d.files}

    # ---- table
    lines = [f'{"run":10s} {"final F>0 %":>12s} {"peak F>0 %":>11s} {"max peak F MPa":>15s} {"mean sqrtJ2 dsig MPa":>21s}']
    for lab, d in R:
        lines.append(f'{lab:10s} {100*(d["finalF"]>0).mean():12.1f} {100*(d["peakF"]>0).mean():11.1f} {d["peakF"].max()/1e6:15.1f} '
                     f'{J[lab].mean() if lab in J else float("nan"):21.2f}')
    lines.append('')
    lines.append(f'agreement with reference "{R[0][0]}" (all receivers):  rel. rms of stress-change tensor | corr of {a.what} F | yield-map agreement')
    for lab, d in R[1:]:
        A, B = ref['dsigma_final'], d['dsigma_final']
        rr = np.sqrt(((A - B)**2).sum() / (A**2).sum())
        c = np.corrcoef(F[R[0][0]].ravel(), F[lab].ravel())[0, 1]
        yy = ((F[R[0][0]] > 0) == (F[lab] > 0)).mean()
        lines.append(f'  {lab:10s} {rr:8.3f} | {c:6.3f} | {100*yy:5.1f} %')
    txt = '\n'.join(lines); print(txt); open(a.out + '_table.txt', 'w').write(txt + '\n')

    # ---- profiles vs depth
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for k, (lab, d) in enumerate(R):
        st = STY[k % 4]
        ax[0].plot((d['finalF'] > 0).mean((0, 1)), z, lw=1.8, label=lab, **st)
        ax[1].plot((d['peakF'] > 0).mean((0, 1)), z, lw=1.8, label=lab, **st)
        if lab in J:
            ax[2].plot(J[lab].mean((0, 1)) / J[R[0][0]].mean((0, 1)), z, lw=1.8, label=lab, **st)
    ax[0].set_title('fraction yielding, final state', loc='left', fontsize=10); ax[1].set_title('fraction yielding, peak over time', loc='left', fontsize=10)
    ax[2].set_title(f'mean stress-change invariant / {R[0][0]}', loc='left', fontsize=10)
    ax[0].set_ylabel('depth (km)'); ax[0].invert_yaxis()
    for x in ax: x.grid(color='#e6e6e6', lw=0.6); x.set_axisbelow(True); [sp.set_visible(False) for k_, sp in x.spines.items() if k_ in ('top', 'right')]
    ax[2].axvline(1, color=MUTE, lw=0.7); ax[0].set_xlim(0, 1.02); ax[1].set_xlim(0, 1.02)
    ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig(a.out + '_profiles.png', dpi=150); plt.close(fig)

    # ---- scatter of stress-change invariant
    if len(R) > 1 and J:
        fig, axs = plt.subplots(1, len(R) - 1, figsize=(4.8 * (len(R) - 1), 4.4), squeeze=False)
        Zd = np.broadcast_to(z[None, None, :], J[R[0][0]].shape)
        for k, (lab, d) in enumerate(R[1:]):
            x = axs[0, k]; sc = x.scatter(J[R[0][0]].ravel(), J[lab].ravel(), c=Zd.ravel(), s=2, cmap='viridis_r', rasterized=True)
            m = max(J[R[0][0]].max(), J[lab].max()); x.plot([0, m], [0, m], color=MUTE, lw=0.7)
            x.set_xlabel(f'{R[0][0]}: sqrt(J2) of stress change (MPa)'); x.set_ylabel(f'{lab}: sqrt(J2) of stress change (MPa)')
            x.set_aspect('equal'); [sp.set_visible(False) for k_, sp in x.spines.items() if k_ in ('top', 'right')]
        fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.8).set_label('depth (km)')
        fig.savefig(a.out + '_scatter.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # ---- per-transect panels
    n = len(s)
    ids = [int(v) - 1 for v in a.transects.split(',')] if a.transects else sorted(set(np.linspace(0.08 * n, 0.92 * n, 5).astype(int)))
    oo = np.arange(off.min(), off.max() + 1, 10.0); omin = np.abs(off).min()
    Fi = {lab: interp1d(off, F[lab], axis=1)(oo) for lab in F}
    vmax = a.vmax or max(np.percentile(Fi[l], 99.5) for l in Fi)
    norm = Normalize(0, vmax)
    for i in ids:
        nc = len(R) + (len(R) - 1)
        fig, axs = plt.subplots(1, nc, figsize=(3.3 * nc + 0.8, 3.9), sharey=True)
        for k, (lab, d) in enumerate(R):
            im = ps.section_axes(axs[k], oo, z, Fi[lab][i].T, norm, omin, h=ps.get_h(d)); axs[k].set_title(lab, fontsize=10, loc='left')
        for k, (lab, d) in enumerate(R[1:]):
            x = axs[len(R) + k]; dF = (Fi[lab][i] - Fi[R[0][0]][i]).T
            lim = max(np.abs(dF).max(), 1e-6)
            im2 = x.pcolormesh(oo, z, dF, cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim), shading='nearest')
            x.axvspan(-omin, omin, color='#dddddd', zorder=3); x.axvline(0, color=MUTE, lw=0.5, zorder=4)
            x.set_ylim(z.max() + 0.1, 0); x.set_xlim(oo.min(), oo.max())
            x.set_title(f'{lab} minus {R[0][0]}', fontsize=10, loc='left')
            fig.colorbar(im2, ax=x, shrink=0.8, pad=0.02).set_label('dF (MPa)')
        for x in axs: x.set_xlabel('offset (m)  (- NE | SW +)')
        axs[0].set_ylabel('depth (km)')
        fig.colorbar(im, ax=axs[:len(R)], shrink=0.8, pad=0.01).set_label(f'F (MPa), {a.what};  F>0 outlined')
        fig.suptitle(f'Transect {i+1}  (s = {s[i]:.1f} km from SE end)', x=0.01, ha='left', fontsize=11)
        fig.savefig(f'{a.out}_T{i+1:03d}.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('wrote', a.out + '_profiles.png', a.out + '_scatter.png', f'and transect panels {[i+1 for i in ids]}')


if __name__ == '__main__':
    matplotlib.use('Agg')
    main()
