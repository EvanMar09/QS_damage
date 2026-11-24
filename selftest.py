"""
selftest.py - quick checks that the installation works and the solvers agree.   python selftest.py   (about 1-2 min)
  1  imports
  2  single dislocation patch:  quasi-static full-space patch  vs  point-source (Aki & Richards) static limit
  3  half-space:  traction-free surface; approaches the full-space solution at depth
  4  tiny end-to-end model with all three solvers (needs cutde for qs-*)
  5  new: kernel cache (rerun is identical), time-slice stress, stress-formula settings, slice figures
"""
import os, sys, time, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ok_all = True


def check(name, cond, info=''):
    global ok_all
    ok_all &= bool(cond)
    print(f'  [{"PASS" if cond else "FAIL"}] {name} {info}')


print('1. imports')
import kinstress as ks
import elsinore_yield as ey
try:
    import cutde.halfspace as H, cutde.fullspace as F
    import qs_stress as qs
    have_cutde = True
except Exception as e:                                   # noqa
    have_cutde = False
    print('  cutde not available (pip install cutde): the quasi-static solvers are skipped.', e)
check('core modules import', True)

med = ks.medium(6000., 3464., 2700.); mu, lam = med[4], med[3]; nu = lam / (2 * (lam + mu))
if have_cutde:
    print('2. one patch: finite-patch full-space vs point-source static limit (distance >= 6 cell sizes)')
    ang = 40; t = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]); n = np.array([-t[1], t[0]])
    hs = 100.; D = 1.0; z0 = 1500.
    a = np.array([-hs / 2 * t[0], -hs / 2 * t[1]]); b = np.array([hs / 2 * t[0], hs / 2 * t[1]])
    tris = np.array([[[*a, -(z0 - hs / 2)], [*b, -(z0 - hs / 2)], [*b, -(z0 + hs / 2)]],
                     [[*a, -(z0 - hs / 2)], [*b, -(z0 + hs / 2)], [*a, -(z0 + hs / 2)]]])
    rec = np.array([[*(600 * t + 400 * n), -1200.], [*(-300 * t + 900 * n), -1800.], [*(1500 * t - 200 * n), -1500.]])
    src = np.array([[0, 0, -z0, t[0], t[1], mu * hs * hs * D, 0.0, 0.05]])
    Sp = ks.stress_history_flat(rec, src, med, 0.05, 600)[:, -1, :]
    K = F.strain_matrix(rec, tris, nu)
    e = np.einsum('oiaj,aj->oi', K, np.tile([[-D, 0, 0]], (2, 1)))
    Sq = F.strain_to_stress(e, mu, nu)[:, [0, 1, 2, 3, 5, 4]]
    err = np.abs(Sq - Sp).max() / np.abs(Sp).max()
    check('patch vs point source', err < 0.03, f'(max relative difference {100*err:.2f} %)')

    print('3. half-space')
    tri = np.array([[[0, 0, 0], [100, 0, 0], [100, 0, -100]], [[0, 0, 0], [100, 0, -100], [0, 0, -100]]], float)
    obs = np.array([[30., 200., 0.], [70., -300., 0.]])
    Kh = H.strain_matrix(obs, tri, nu); eh = (Kh[:, :, :, 0] * -1).sum(2)
    sh = H.strain_to_stress(eh, mu, nu)
    check('traction-free surface (sigma_xz, sigma_yz = 0)', np.abs(sh[:, [4, 5]]).max() < 1e-3 * np.abs(sh).max(),
          f'(max {np.abs(sh[:, [4, 5]]).max():.2e} Pa)')
    deep = np.array([[300., 200., -9000.]]); tri_d = tri.copy(); tri_d[..., 2] -= 9000.
    e1 = (H.strain_matrix(deep, tri_d, nu)[:, :, :, 0]).sum(2); e2 = (F.strain_matrix(deep, tri_d, nu)[:, :, :, 0]).sum(2)
    check('half-space -> full-space at 9 km depth', np.abs(e1 - e2).max() / np.abs(e2).max() < 0.02,
          f'(difference {100*np.abs(e1-e2).max()/np.abs(e2).max():.2f} %)')

print('4. tiny model, all solvers (h = 400 m)')
geo = 'Elsinore_simple30m_smooth1.geo'
if not os.path.exists(geo):
    print('  geo file not found next to selftest.py, skipping'); ok_all = False
else:
    tmp = tempfile.mkdtemp()
    base = ['--geo', geo, '--max-slip', '3', '--vr', '2700', '--shallow-frac', '1.0', '--hypo', 'M', '--h', '400', '--dz', '1000',
            '--ds', '3000', '--offsets', '1200,2000', '--tmax', '10', '--dt', '0.25']
    res = {}
    import io, contextlib
    for s in (['ar', 'qs-full', 'qs-half'] if have_cutde else ['ar']):
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            ey.main(base + ['--solver', s, '--out', os.path.join(tmp, s)])
        d = np.load(os.path.join(tmp, s + '.npz')); res[s] = d
        check(f'{s}: finite output', np.isfinite(d['finalF']).all() and np.isfinite(d['dsigma_final']).all(),
              f'(final yield {100*(d["finalF"]>0).mean():.1f} %, {time.time()-t0:.0f} s)')
    if have_cutde:
        A, B = res['ar']['dsigma_final'], res['qs-full']['dsigma_final']
        r = np.sqrt(((A - B) ** 2).sum() / (B ** 2).sum())
        check('dynamic vs quasi-static (full space) stress change', r < 0.15, f'(relative rms {100*r:.1f} %, offsets >= 3h)')
        hm, fm = res['qs-half']['dsigma_final'], B
        z = res['ar']['depth_rec']
        sh = (np.sqrt((hm ** 2).sum(-1)).mean((0, 1))); sf = (np.sqrt((fm ** 2).sum(-1)).mean((0, 1)))
        check('free surface reduces the shallow stress change', (sh / sf)[0] < 0.8 and (sh / sf)[-1] > 0.9,
              f'(half/full = {(sh/sf)[0]:.2f} at {z[0]/1e3:.1f} km, {(sh/sf)[-1]:.2f} at {z[-1]/1e3:.1f} km)')

if have_cutde and os.path.exists(geo):
    print('5. kernel cache, time slices, stress formula settings')
    cd = os.path.join(tmp, 'kc')
    sl = base + ['--solver', 'qs-half', '--save-times', '4,8', '--kernel-cache-dir', cd]
    with contextlib.redirect_stdout(io.StringIO()):
        ey.main(sl + ['--out', os.path.join(tmp, 'k1')])
        t0 = time.time(); ey.main(sl + ['--out', os.path.join(tmp, 'k2')]); t_cached = time.time() - t0
        ey.main(base + ['--solver', 'qs-half', '--save-times', '4,8', '--no-kernel-cache', '--out', os.path.join(tmp, 'k3')])
    k1, k2, k3 = (np.load(os.path.join(tmp, f'k{i}.npz')) for i in (1, 2, 3))
    check('kernel cache file written', len([f for f in os.listdir(cd) if f.startswith('qs_kernel_')]) == 1)
    e = np.abs(k2['dsigma_final'] - k3['dsigma_final']).max() / np.abs(k3['dsigma_final']).max()
    check('cached rerun equals uncached run', e < 1e-5, f'(relative difference {e:.1e}, cached rerun {t_cached:.1f} s)')
    Mx, Fn = k2['Fmax_slices'], k2['F_slices']
    check('running maximum of F at slice times: >= F now, never decreases, <= peak', bool((Mx >= Fn - 1e-3).all() and (np.diff(Mx, axis=-1) >= -1e-3).all()
          and (Mx <= k2['peakF'][..., None] + 1e-3).all()))
    check('slice stress tensors saved (2 times x 6 components), finite', k2['dsigma_slices'].shape[-2:] == (2, 6)
          and np.isfinite(k2['dsigma_slices']).all(), f'(shape {k2["dsigma_slices"].shape})')
    import plot_slices as psl, matplotlib
    matplotlib.use('Agg')
    out = psl.make_slices(os.path.join(tmp, 'k2.npz'), transects='2', dpi=60, verbose=False)
    nf = sum(len(f) for _, _, f in os.walk(out))
    check('slice figures written (2 times x (plan view + yield + normal% + mean%))', nf == 8, f'({nf} files)')
    ey.set_stress_params(0.30, 5000.0, 0.3652, None)
    z1 = ey.initial_stress(np.array([2000.0]), 350.0); ey.reset_stress_params()
    z0 = ey.initial_stress(np.array([2000.0]), 350.0)
    check('LITHO_COEF scales the initial stress (0.30 / 0.25 = 1.2)', abs(z1[0, :3].mean() / z0[0, :3].mean() - 1.2) < 1e-6)

sel_ok = (list(ey.select_transects('every:5', 50) + 1) == [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
          and list(ey.select_transects('n:10', 50) + 1)[0::9] == [1, 50] and len(ey.select_transects('n:10', 50)) == 10
          and list(ey.select_transects('3,7', 50) + 1) == [3, 7] and len(ey.select_transects('all', 50)) == 50)
check('transect subset selection (every:k, n:m, list, all)', sel_ok)

# rise time after Graves & Pitarka (2016): fault average pinned to the moment scaling, rise ~ sqrt(slip) when eta = 1 and no depth factor, x2 in the shallow zone
import scenario_stoch as sst
_z = (np.arange(150) + 0.5) * 100.0
_D = 0.5 + 3.0 * np.random.default_rng(3).random((200, 150)) ** 2
_M0 = 3.2e10 * 1e4 * _D.sum()
_T1, _i1 = sst.gp_rise_time(_D, _z, _M0, eta=1.0, depth_factor=False)
_T2, _i2 = sst.gp_rise_time(_D, _z, _M0, eta=1.0, depth_factor=True)
_act = _D >= 0.1 * _D.max()
_ok = (abs(_i1['mean'] - 1.6e-9 * (_M0 * 1e7) ** (1 / 3)) < 1e-6 * _i1['tau_A'] + 1e-3 and np.corrcoef(np.sqrt(_D[_act]), _T1[_act])[0, 1] > 0.98
       and abs(_i2['mean'] - _i1['tau_A']) < 1e-3 and _T2[:, :40].mean() > 1.5 * _T2[:, 90:140].mean())
check('Graves & Pitarka rise time (moment-scaled mean, sqrt(slip), shallow x2)', _ok, f"(mean {_i1['mean']:.2f} s, corr {np.corrcoef(np.sqrt(_D[_act]), _T1[_act])[0, 1]:.3f})")

# slip taper switches
_S, _Z = np.meshgrid(np.arange(50.0) * 100 + 50, np.arange(40.0) * 100 + 50, indexing='ij')
_w1 = sst.taper_weights(_S, _Z, 5000.0, 4000.0, 3000.0, 2000.0); _w0 = sst.taper_weights(_S, _Z, 5000.0, 4000.0, 0.0, 0.0)
check('slip taper can be switched off (TAPER_S / TAPER_Z = 0)', _w1[0][0, 0] < 0.01 and _w1[1][0, -1] < 0.01 and np.all(_w0[0] == 1) and np.all(_w0[1] == 1))

# optional: SeisSol yaml stress reader (needs lupa and the yaml in this folder)
try:
    import lupa  # noqa
    ymlf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fault_tensor_depthdep_TW_S.yaml')
    if os.path.exists(ymlf):
        zz = np.array([0., 500., 1000., 2000., 5000., 12000.])
        ref = ey.initial_stress(zz, 350.0)
        ey.use_seissol_yaml(ymlf)
        got = ey.initial_stress(zz)
        ey.reset_stress_params()
        check('SeisSol yaml stress equals the built-in stress (this yaml)', np.abs(got - ref).max() < 1.0,
              f'(max difference {np.abs(got - ref).max():.1e} Pa)')
except ImportError:
    print('  [skip] SeisSol yaml reader (pip install lupa to enable)')
print('\nALL CHECKS PASSED' if ok_all else '\nSOME CHECKS FAILED')
sys.exit(0 if ok_all else 1)
