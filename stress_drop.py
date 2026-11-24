#!/usr/bin/env python3
"""
stress_drop.py - static stress drop on the fault for a saved run.

  python stress_drop.py run.npz [--solver qs-full|qs-half|ar] [--delta 50] [--step 3] [--geo ...]

Evaluates the final (static) shear-stress change resolved along strike at +-delta from the fault, averages both
sides, prints statistics, saves <run>_stress_drop.png and <run>_stress_drop.npz.

  --solver qs-full (default)  exact finite-patch static solution, full space.  delta default 0.5 h.
  --solver qs-half            same with the free surface (needs cutde; ~8x slower)
  --solver ar                 point-source solution used by the dynamic solver; delta default 1.5 h, ~25 % accurate there
"""
import sys, argparse, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import elsinore_yield as ey

ap = argparse.ArgumentParser()
ap.add_argument('npz'); ap.add_argument('--geo', default='Elsinore_simple30m_smooth1.geo')
ap.add_argument('--delta', type=float, default=None, help='evaluation distance from fault (m); default 1.5*h')
ap.add_argument('--step', type=int, default=None, help='grid subsampling relative to the source grid (default 3 for qs, 2 for ar)')
ap.add_argument('--solver', default='qs-full', choices=['qs-full', 'qs-half', 'ar'])
a = ap.parse_args(sys.argv[1:])
d = np.load(a.npz)
trace, med, fine, coarse, h, p = ey.rebuild_from_npz(d, a.geo)
ey.restore_stress_from_params(p)          # use the SeisSol-yaml stress if the run used it
a.step = a.step or (2 if a.solver == 'ar' else 3)
delta = a.delta or (1.5 * h if a.solver == 'ar' else 0.5 * h)
s_map = d['s_src'][::a.step]; z_map = d['z_src'][::a.step]
if a.solver == 'ar':
    dtau, tau0, sn0 = ey.static_stress_drop(trace, fine, coarse, med, h, float(d['W']), p['eps_frac'] * h,
                                            3 * p['blocks'] * h, s_map, z_map, delta, shmax=p['shmax'])
else:
    import qs_stress as qs
    P_, Tg, N = trace.pos(s_map), trace.tang(s_map), trace.normal(s_map)
    xy = np.stack([P_ - delta * N, P_ + delta * N], 1)                                   # [ns,2,2]
    grid = np.empty((len(s_map), 2, len(z_map), 3)); grid[..., :2] = xy[:, :, None, :]; grid[..., 2] = -z_map[None, None, :]
    nu = med[3] / (2 * (med[3] + med[4]))
    St = qs.qs_history(grid, s_map, z_map, trace, d['s_src'], d['z_src'], d['slip'], d['rt'], d['rise'], np.array([1e6]),
                       med[4], nu, mode='full' if a.solver == 'qs-full' else 'half', verbose=False)[:, 0, :].reshape(len(s_map), 2, len(z_map), 6)
    TG, NM = Tg[:, None, :], N[:, None, :]
    dts = (TG[..., 0] * NM[..., 0])[..., None] * St[..., 0] + (TG[..., 1] * NM[..., 1])[..., None] * St[..., 1] \
        + ((TG[..., 0] * NM[..., 1] + TG[..., 1] * NM[..., 0]))[..., None] * St[..., 3]
    dtau = dts.mean(1)
    S_, Z_ = np.meshgrid(np.arange(len(s_map)), z_map, indexing='ij')
    i0 = ey.initial_stress(Z_, p['shmax'])
    Tm, Nm = Tg[S_], N[S_]
    tau0 = Tm[..., 0] * Nm[..., 0] * i0[..., 0] + Tm[..., 1] * Nm[..., 1] * i0[..., 1] + (Tm[..., 0] * Nm[..., 1] + Tm[..., 1] * Nm[..., 0]) * i0[..., 3]
    sn0 = -(Nm[..., 0] ** 2 * i0[..., 0] + Nm[..., 1] ** 2 * i0[..., 1] + 2 * Nm[..., 0] * Nm[..., 1] * i0[..., 3])
D = d['slip'][::a.step, ::a.step]
st = ey.stress_drop_summary(D, dtau, h * a.step, med[4])
tau_end = tau0 + dtau
print(f"solver {a.solver}, evaluated at +-{delta:.0f} m from the fault (h = {h:.0f} m, grid step {a.step})")
print(f"rupture area (slip > 10% of max): {st['area_km2']:.0f} km2, mean slip {st['mean_slip']:.2f} m, M0 = {st['M0']:.2e} N m")
print(f"stress drop, area mean        : {st['mean_drop_area']/1e6:.2f} MPa")
print(f"stress drop, slip-weighted    : {st['mean_drop_slipweighted']/1e6:.2f} MPa")
print(f"Eshelby estimate 7/16 M0/a^3  : {st['eshelby_drop']/1e6:.2f} MPa  (a = {st['a_eq']/1e3:.1f} km equivalent radius)")
m = D > 0.1 * D.max()
print(f"initial shear stress on fault (slipping area): mean {tau0[m].mean()/1e6:.2f} MPa;  final: mean {tau_end[m].mean()/1e6:.2f} MPa;  "
      f"fraction of cells with final tau<0 (overshoot): {(tau_end[m] < 0).mean()*100:.1f}%")
S_, Z_ = np.meshgrid(s_map, z_map, indexing='ij')
core = (Z_ > 2000) & (Z_ < float(d['W']) - 4000) & (S_ > 3000) & (S_ < float(d['L']) - 3000) & m
print(f"core region (depth 2 to {(float(d['W'])-4000)/1e3:.0f} km, >3 km from the ends): mean drop {(-dtau[core]).mean()/1e6:.2f} MPa, "
      f"median {np.median(-dtau[core])/1e6:.2f} MPa, 10-90%: {np.percentile(-dtau[core],10)/1e6:.1f} to {np.percentile(-dtau[core],90)/1e6:.1f} MPa")
np.savez_compressed(a.npz.replace('.npz', '_stress_drop.npz'), s=s_map, z=z_map, dtau=dtau, tau0=tau0, sn0=sn0, **st)
fig, ax = plt.subplots(figsize=(11, 3.8))
v = np.percentile(np.abs(dtau[m]), 98) / 1e6
im = ax.pcolormesh(s_map / 1e3, z_map / 1e3, -dtau.T / 1e6, cmap='RdBu_r', vmin=-v, vmax=v, shading='nearest')
ax.set_ylim(float(d['W']) / 1e3, 0); ax.set_xlabel('along-strike distance from SE end (km)'); ax.set_ylabel('depth (km)')
fig.colorbar(im, ax=ax, pad=0.01).set_label('static stress drop (MPa); blue = stress increase')
ax.set_title(f"Static stress drop: area mean {st['mean_drop_area']/1e6:.1f} MPa", loc='left', fontsize=9.5)
fig.savefig(a.npz.replace('.npz', '_stress_drop.png'), dpi=160, bbox_inches='tight')
