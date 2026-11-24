#!/usr/bin/env python3
"""plot_transects.py run.npz : fault-normal cross sections (offset vs depth) on each transect.
Writes <run>_transects_sheet.png (all transects, peak F) and <run>_transects.pdf (one page each: peak & final F)."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d

d = np.load(sys.argv[1]); base = sys.argv[1].replace('.npz', '')
s = d['s_rec'] / 1e3; off = d['offsets']; z = d['depth_rec'] / 1e3
F = d['peakF'] / 1e6; Ff = d['finalF'] / 1e6                 # (transect, offset, depth)
nt = F.shape[0]
omin = np.abs(off).min()
# interpolate to a uniform offset axis for smooth sections; the strip |offset|<omin is not resolved
oo = np.arange(off.min(), off.max() + 1, 10.0)
def reg(A):
    out = interp1d(off, A, axis=1, kind='linear')(oo)
    return out
Fr, Ffr = reg(F), reg(Ff)
vmax = max(np.percentile(F[F > 0], 98) if (F > 0).any() else 1.0, 1.0)
norm = Normalize(0, vmax); cm = 'Reds'; INK, MUTE = '#222222', '#777777'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTE, 'axes.spines.top': False, 'axes.spines.right': False})

def panel(a, A, title=None):
    im = a.pcolormesh(oo, z, np.maximum(A[i].T if False else A, 0), cmap=cm, norm=norm, shading='nearest')
    a.contour(oo, z, A, levels=[0], colors=INK, linewidths=0.7)
    a.axvspan(-omin, omin, color='#dddddd', zorder=3)
    a.axvline(0, color=MUTE, lw=0.5, zorder=4)
    a.set_ylim(z.max() + 0.1, 0); a.set_xlim(oo.min(), oo.max())
    return im

# contact sheet
nc = 5; nr = int(np.ceil(nt / nc))
fig, ax = plt.subplots(nr, nc, figsize=(14, 2.1 * nr), sharex=True, sharey=True)
for i, a in enumerate(ax.ravel()):
    if i >= nt: a.axis('off'); continue
    im = panel(a, Fr[i].T)
    a.set_title(f'T{i+1}  s={s[i]:.1f} km', fontsize=7.5, loc='left', color=MUTE, pad=2)
for a in ax[:, 0]: a.set_ylabel('depth (km)')
for a in ax[-1]: a.set_xlabel('offset (m, - NE / + SW)')
fig.subplots_adjust(right=0.9, hspace=0.35)
cax = fig.add_axes([0.92, 0.3, 0.012, 0.4]); fig.colorbar(im, cax=cax).set_label('peak F (MPa); F>0 outlined; gray = unresolved')
fig.savefig(base + '_transects_sheet.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# pdf, one page per transect
slip = d['slip']; ss = d['s_src'] / 1e3
with PdfPages(base + '_transects.pdf') as pdf:
    for i in range(nt):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for a, A, t in ((ax[0], Fr[i].T, 'peak over time'), (ax[1], Ffr[i].T, 'final state')):
            im = panel(a, A); a.set_title(t, fontsize=9, loc='left'); a.set_xlabel('offset from fault (m)   (- NE | SW +)')
        ax[0].set_ylabel('depth (km)')
        j = int(np.argmin(abs(ss - s[i])))
        fig.suptitle(f'Transect {i+1}: s = {s[i]:.1f} km from SE end;  surface-cell slip {slip[j, 0]:.2f} m;  '
                     f'max peak F = {F[i].max():.1f} MPa', fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).set_label('Drucker-Prager exceedance F (MPa)')
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
print('done', base)
