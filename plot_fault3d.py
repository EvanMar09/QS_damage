"""
plot_fault3d.py - 3-D view of the (vertical, curved-trace) fault: slip in color, rupture-time contours as lines,
hypocenter as a star, and a dot at the surface for every cross-section.

  python plot_fault3d.py run.npz                       # run_fault3d.png  (+ run_fault.vtk for ParaView)
  python plot_fault3d.py run.npz --elev 20 --azim -60 --vexag 2
Spyder:  import plot_fault3d as pf; fig = pf.plot_fault_3d(np.load('run.npz'), show=True)   # with %matplotlib qt you can rotate it
"""
import argparse, numpy as np, matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

INK, MUTE = '#222222', '#777777'


def fault_xyz(d, s, z, lift=0.0):
    """3-D km coordinates (east, north, up) of fault points at arc-length s (m) and depth z (m).  `lift` (m) pushes the
    points off the fault toward its SW side (toward a viewer standing SW), to keep lines from sinking into the surface."""
    tr = np.asarray(d['trace'], float); L = float(d['L'])
    sa = np.linspace(0, L, len(tr))
    x = np.interp(s, sa, tr[:, 0]); y = np.interp(s, sa, tr[:, 1])
    if lift:
        dx = np.interp(s, sa, np.gradient(tr[:, 0], sa)); dy = np.interp(s, sa, np.gradient(tr[:, 1], sa))
        n = np.hypot(dx, dy); nx, ny = -dy / n, dx / n          # left normal of the SE->NW tangent = SW side
        x = x + lift * nx; y = y + lift * ny
    return x / 1e3, y / 1e3, -np.asarray(z, float) / 1e3


def plot_fault_3d(d, out=None, contour_dt=1.0, elev=18, azim=None, vexag=1.0, stride=2, show=False, cmap='YlOrBr', slip_max=None):
    s = d['s_src']; z = d['z_src']; D = d['slip']; rt = d['rt']; L = float(d['L']); W = float(d['W'])
    si = np.arange(0, len(s), stride); zi = np.arange(0, len(z), stride)
    S, Z = np.meshgrid(s[si], z[zi], indexing='ij')
    X, Y, Zk = fault_xyz(d, S.ravel(), Z.ravel()); X, Y, Zk = (a.reshape(S.shape) for a in (X, Y, Zk))
    norm = Normalize(0, D.max() if slip_max is None else slip_max)   # slip_max: common color-scale top (m) for comparing runs
    fc = plt.get_cmap(cmap)(norm(D[np.ix_(si, zi)]))

    fig = plt.figure(figsize=(11, 6.2)); ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Zk, facecolors=fc, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

    # rupture-time contours as 3-D lines slightly off the fault
    lv = np.arange(contour_dt, np.ceil(rt[D > 0.05 * D.max()].max()) + contour_dt, contour_dt)
    f2 = plt.figure(); a2 = f2.add_subplot(111); cs = a2.contour(s, z, rt.T, levels=lv); plt.close(f2)
    for lev, segs in zip(cs.levels, cs.allsegs):
        for sg in segs:
            if len(sg) < 2: continue
            x, y, zz = fault_xyz(d, sg[:, 0], sg[:, 1], lift=40.0)
            ax.plot(x, y, zz, color=INK, lw=0.9, zorder=10)
            k = int(np.argmin(sg[:, 1]))                              # label where the contour is shallowest
            ax.text(x[k], y[k], zz[k] + 0.3, f'{lev:g} s', fontsize=6, color=INK, ha='center', zorder=11)

    hs, hz = float(d['hypo_s']), float(d['hypo_depth'])
    x, y, zz = fault_xyz(d, np.array([hs]), np.array([hz]), lift=60.0)
    ax.plot(x, y, zz, '*', color='white', mec=INK, ms=15, mew=1, zorder=12)
    if 's_rec' in d.files:
        x, y, zz = fault_xyz(d, d['s_rec'], np.zeros_like(d['s_rec']), lift=40.0)
        ax.plot(x, y, zz, 'o', ms=2.2, color=INK, zorder=12)

    # view from the SW side (looks NE): NW on the left, SE on the right
    tr = np.asarray(d['trace'], float); v = tr[0] - tr[-1]
    az = (np.degrees(np.arctan2(v[1], v[0])) - 90) if azim is None else azim      # camera looks across the fault from SW
    ax.view_init(elev=elev, azim=az)
    ext = [np.ptp(X), np.ptp(Y), np.ptp(Zk) * vexag]
    ax.set_box_aspect(ext)
    ax.set_xlabel('east (km, local)'); ax.set_ylabel('north (km, local)'); ax.set_zlabel('depth (km)')
    zt = np.arange(0, W / 1e3 + 1, 5); ax.set_zticks(-zt); ax.set_zticklabels([f'{t:g}' for t in zt])
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.set_facecolor((1, 1, 1, 0)); a_.pane.set_edgecolor('#cccccc')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02).set_label('final slip (m)')
    ax.set_title(f'Fault slip with rupture-time contours every {contour_dt:g} s  (star = hypocenter, dots = cross-sections)',
                 loc='left', fontsize=9.5)
    if out:
        fig.savefig(out, dpi=170, bbox_inches='tight')
    if show:
        plt.show()
    elif out:
        plt.close(fig)
    return fig


def write_fault_vtk(d, path):
    """slip, rupture time, rise time on the fault surface for ParaView."""
    import elsinore_yield as ey
    S, Z = np.meshgrid(d['s_src'], d['z_src'], indexing='ij')
    x, y, zz = fault_xyz(d, S.ravel(), Z.ravel())
    P = np.stack([x * 1e3, y * 1e3, zz * 1e3], -1).reshape(S.shape + (3,))[:, None, :, :]
    f = {'slip_m': d['slip'][:, None, :], 'rupture_time_s': d['rt'][:, None, :], 'rise_time_s': d['rise'][:, None, :]}
    ey.write_vtk(path, P, f)


if __name__ == '__main__':
    matplotlib.use('Agg')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz'); ap.add_argument('--contour-dt', type=float, default=1.0)
    ap.add_argument('--elev', type=float, default=18); ap.add_argument('--azim', type=float, default=None)
    ap.add_argument('--vexag', type=float, default=1.0); ap.add_argument('--slip-max', type=float, default=None); ap.add_argument('--no-vtk', action='store_true')
    a = ap.parse_args(); d = np.load(a.npz); base = a.npz.replace('.npz', '')
    plot_fault_3d(d, base + '_fault3d.png', a.contour_dt, a.elev, a.azim, a.vexag, slip_max=a.slip_max); print('wrote', base + '_fault3d.png')
    if not a.no_vtk:
        write_fault_vtk(d, base + '_fault.vtk'); print('wrote', base + '_fault.vtk')
