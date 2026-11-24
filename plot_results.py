#!/usr/bin/env python3
"""plot_results.py <run.npz> : yield-potential figures from elsinore_yield.py output."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

d = np.load(sys.argv[1]); base = sys.argv[1].replace('.npz', '')
s = d['s_rec'] / 1e3; off = d['offsets']; z = d['depth_rec'] / 1e3
F = d['peakF'] / 1e6; Ff = d['finalF'] / 1e6          # (s, offset, depth)
L = float(d['L']) / 1e3
INK, MUTE = '#222222', '#777777'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTE, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})
vmax = max(np.percentile(F[F > 0], 98) if (F > 0).any() else 1, 1)
norm = Normalize(0, vmax)
cm = 'Reds'

# ---- 1. fault-parallel sections (s vs depth) at each offset, like Ma & Andrews Fig 8b
nof = len(off)
fig, ax = plt.subplots(nof, 1, figsize=(11, 1.9 * nof + 1), sharex=True, sharey=True)
for k, o in enumerate(off):
    a = ax[k]
    im = a.pcolormesh(s, z, np.maximum(F[:, k, :].T, 0), cmap=cm, norm=norm, shading='nearest')
    a.contour(s, z, F[:, k, :].T, levels=[0], colors=INK, linewidths=0.8)
    hs = d['hypo_s'] / 1e3
    a.plot(hs, float(d['hypo_depth']) / 1e3, '*', color=INK, ms=9, mfc='white')
    a.set_ylabel('depth (km)'); a.invert_yaxis()
    a.text(0.995, 0.06, f'offset {o:+.0f} m ({"SW" if o>0 else "NE"})', transform=a.transAxes, ha='right', fontsize=8, color=MUTE)
ax[-1].set_xlabel('along-strike distance from SE end (km)')
cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8); cb.set_label('peak Drucker-Prager exceedance F (MPa), F>0 outlined')
fig.suptitle('Kinematic yield potential: fault-parallel sections (peak over time)', x=0.45)
fig.savefig(base + '_sections.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---- 2. unfolded map view at depth slices: along-strike vs fault-normal offset
depths = [1.5, 3.5, 6.0, 8.0, 11.0]
fig, ax = plt.subplots(len(depths), 1, figsize=(11, 1.7 * len(depths) + 1), sharex=True)
for a, dz_ in zip(ax, depths):
    j = int(np.argmin(abs(z - dz_)))
    X = np.r_[s]; yy = off
    Z_ = F[:, :, j].T
    # offsets are not evenly spaced: use pcolormesh on edges from midpoints
    e = np.r_[off[0] - (off[1] - off[0]) / 2, (off[1:] + off[:-1]) / 2, off[-1] + (off[-1] - off[-2]) / 2]
    im = a.pcolormesh(s, off, np.maximum(Z_, 0), cmap=cm, norm=norm, shading='nearest')
    a.contour(s, off, Z_, levels=[0], colors=INK, linewidths=0.8)
    a.axhline(0, color=MUTE, lw=0.6)
    a.set_ylabel('offset (m)'); a.set_title(f'depth {z[j]:.2f} km', loc='left', fontsize=8.5, color=MUTE)
ax[-1].set_xlabel('along-strike distance from SE end (km)')
cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8); cb.set_label('peak F (MPa)')
fig.suptitle('Unfolded map view (NE side = negative offset, SW side = positive)', x=0.45)
fig.savefig(base + '_mapslices.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---- 3. true map view: trace + along-strike envelope of yield potential (max over depth, nearest offset)
rec = d['rec']; tr = d['trace'] / 1e3
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.plot(tr[:, 0], tr[:, 1], color=INK, lw=1)
Fmax = np.maximum(F.max(2), 0)                          # max over depth (s, offset)
xs, ys, cs = rec[:, :, 0, 0] / 1e3, rec[:, :, 0, 1] / 1e3, Fmax
m = cs > 0
sc = ax.scatter(xs[m], ys[m], c=cs[m], cmap=cm, norm=norm, s=14, zorder=3)
ax.scatter(xs[~m], ys[~m], c='#cccccc', s=5, zorder=2)
ax.set_aspect('equal'); ax.set_xlabel('local x, east (km)'); ax.set_ylabel('local y, north (km)')
cb = fig.colorbar(sc, ax=ax, shrink=0.7); cb.set_label('max over depth of peak F (MPa)')
ax.set_title('Map view: yield potential (gray = no yielding at any depth)')
fig.savefig(base + '_map.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---- 4. scenario + diagnostics: slip, rupture time, static on-fault stress change
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
ss = d['s_src'] / 1e3; zs = d['z_src'] / 1e3
for a, v, t, c in ((ax[0], d['slip'], 'final slip (m)', 'Grays'), (ax[1], d['rt'], 'rupture time (s)', 'viridis')):
    im = a.pcolormesh(ss, zs, v.T, cmap=c, shading='nearest'); a.invert_yaxis(); a.set_ylabel('depth (km)')
    fig.colorbar(im, ax=a, pad=0.01).set_label(t)
    a.plot(d['hypo_s'] / 1e3, float(d['hypo_depth']) / 1e3, '*', color='#d62728', ms=10, mfc='white')
dt_ = d['dtau_fault'].mean(1) / 1e6
im = ax[2].pcolormesh(s, z, dt_.T, cmap='Blues_r', shading='nearest'); ax[2].invert_yaxis()
fig.colorbar(im, ax=ax[2], pad=0.01).set_label('static shear-stress change at fault (MPa)')
ax[2].set_ylabel('depth (km)'); ax[2].set_xlabel('along-strike distance from SE end (km)')
fig.savefig(base + '_scenario.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---- 5. peak vs final
print('fraction F>0 (peak): %.1f%%, (final): %.1f%%' % (100 * (F > 0).mean(), 100 * (Ff > 0).mean()))
for k, o in enumerate(off):
    print(f'offset {o:+5.0f} m: peak F>0 at {100*(F[:,k,:]>0).mean():.0f}% of points, max {F[:,k,:].max():.1f} MPa')
