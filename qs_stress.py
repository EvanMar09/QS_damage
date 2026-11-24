"""
qs_stress.py - quasi-static stress history from prescribed slip, using exact dislocation solutions.

Every subfault is a finite rectangular vertical dislocation patch (two triangles).  Stress at a receiver is the sum of
the static stress of all patches with the slip they have reached at that time, slip(t) = D * raised cosine from
rupture time t0 over rise time T.  No waves: the medium responds instantly (infinite wave speed), so rupture speed
only enters through the slip timing.

  mode 'full' : infinite homogeneous elastic space                 (Nikkhoo & Walter 2015, triangular dislocations)
  mode 'half' : homogeneous elastic half-space with a free surface (same method, free-surface version; the exact
                                                                      half-space solution of the Okada type)
Implemented with the `cutde` package (pip install cutde).

Efficiency: a three-level hierarchy of patches (cells, 5x5 blocks, 25x25 blocks) is used per receiver group; a block
replaces its cells when it is >= theta block-sizes away (default 2.5), with the mean slip history of its cells.

Stress component order everywhere: xx yy zz xy yz xz  (compression negative, Pa).
"""
import time
import numpy as np
import numba as nb
import cutde.halfspace as _H
import cutde.fullspace as _F


def slip_snapshots(D, t0, T, times):
    """slip history [ns,nz,nt] (float32): D * 0.5*(1-cos(pi*(t-t0)/T)) clipped to [0, D]."""
    x = np.clip((times[None, None, :] - t0[..., None]) / T[..., None], 0.0, 1.0)
    return (D[..., None] * 0.5 * (1.0 - np.cos(np.pi * x))).astype(np.float32)


def _block_mean(S, f):
    ns, nz, nt = S.shape
    i = np.arange(0, ns, f); j = np.arange(0, nz, f)
    A = np.add.reduceat(np.add.reduceat(S, i, axis=0), j, axis=1)
    ci = np.diff(np.r_[i, ns]); cj = np.diff(np.r_[j, nz])
    return A / (ci[:, None] * cj[None, :])[..., None], i, np.r_[i[1:], ns], j, np.r_[j[1:], nz]


class _Level:
    """patches of one level: tris[P,2,3,3], center (s,z), size, slip history S[P,nt], parent indices"""
    def __init__(self, trace, ds, dz, i0, i1, j0, j1, S):
        gi0, gj0 = np.meshgrid(i0, j0, indexing='ij'); gi1, gj1 = np.meshgrid(i1, j1, indexing='ij')
        self.shape = gi0.shape
        gi0, gi1, gj0, gj1 = (a.ravel() for a in (gi0, gi1, gj0, gj1))
        s0, s1 = gi0 * ds, gi1 * ds
        z0, z1 = gj0 * dz, gj1 * dz
        a = trace.pos(s0); b = trace.pos(s1)
        P = len(s0)
        p00 = np.c_[a, -z0]; p10 = np.c_[b, -z0]; p11 = np.c_[b, -z1]; p01 = np.c_[a, -z1]
        tr = np.empty((P, 2, 3, 3))
        tr[:, 0, 0], tr[:, 0, 1], tr[:, 0, 2] = p00, p10, p11
        tr[:, 1, 0], tr[:, 1, 1], tr[:, 1, 2] = p00, p11, p01
        self.tris = tr
        self.sc = 0.5 * (s0 + s1); self.zc = 0.5 * (z0 + z1)
        self.size = np.maximum(s1 - s0, z1 - z0)
        self.S = S.reshape(P, -1)
        self.gi0, self.gj0 = gi0, gj0


def _geometry(trace, s_src, z_src, D_shape, f1, f2):
    """Hierarchy geometry (independent of the slip): three levels of patches and the parent maps."""
    ns, nz = D_shape
    ds, dz = (s_src[1] - s_src[0]), (z_src[1] - z_src[0])
    dummy = np.zeros((ns, nz, 1), np.float32)
    lv0 = _Level(trace, ds, dz, np.arange(ns), np.arange(1, ns + 1), np.arange(nz), np.arange(1, nz + 1), dummy)
    _, i0, i1, j0, j1 = _block_mean(dummy, f1); lv1 = _Level(trace, ds, dz, i0, i1, j0, j1, np.zeros((len(i0), len(j0), 1), np.float32))
    _, i0, i1, j0, j1 = _block_mean(dummy, f2); lv2 = _Level(trace, ds, dz, i0, i1, j0, j1, np.zeros((len(i0), len(j0), 1), np.float32))
    nb1z = lv1.shape[1]; nb2z = lv2.shape[1]
    ci, cj = np.meshgrid(np.arange(ns), np.arange(nz), indexing='ij'); ci, cj = ci.ravel(), cj.ravel()
    par0_1 = (ci // f1) * nb1z + (cj // f1); par0_2 = (ci // f2) * nb2z + (cj // f2)
    b1i, b1j = np.meshgrid(np.arange(lv1.shape[0]), np.arange(nb1z), indexing='ij')
    par1_2 = ((b1i.ravel() * f1) // f2) * nb2z + (b1j.ravel() * f1) // f2
    return lv0, lv1, lv2, par0_1, par0_2, par1_2


def _kernel_key(rec_grid, s_src, z_src, trace, mu, nu, mode, theta, f1, f2, D_shape):
    import hashlib
    h = hashlib.md5()
    for a in (np.round(rec_grid, 3), np.round(np.asarray(s_src[[0, -1]], float), 3), np.round(np.asarray(z_src[[0, -1]], float), 3),
              trace.pos(np.linspace(0, trace.L, 50)).round(3)):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((mode, round(float(theta), 6), round(float(mu), 3), round(float(nu), 8), f1, f2, tuple(D_shape))).encode())
    return h.hexdigest()[:16]


def qs_screen(rec_grid, s_rec, z_rec, trace, s_src, z_src, D, t0, T, times, mu, nu, init, coh, phi, pf, dt, kidx,
              mode='full', theta=2.5, f1=5, f2=25, cache_dir=None, cache_max_gb=3.0, verbose=True, nsus=1):
    """Quasi-static stress history AND Drucker-Prager screening, one transect at a time (low memory).
    Same results as screen_from_acc(qs_history(...)); returns the 12 arrays of screen_from_acc.

    Kernel cache: the stress of every patch group at every receiver does not depend on the slip, only on the geometry.
    With cache_dir set, it is saved the first time (float32) and reused by every later run on the same geometry
    (a different slip, rupture time, rise time, seed ...), so only the cheap slip combination is redone."""
    import os
    M = _F if mode == 'full' else _H
    ns, nz = D.shape
    nt = len(times); nsr, no, nzr, _ = rec_grid.shape
    S0 = slip_snapshots(D, t0, T, times)
    lv0, lv1, lv2, par0_1, par0_2, par1_2 = _geometry(trace, s_src, z_src, D.shape, f1, f2)
    ds, dz = (s_src[1] - s_src[0]), (z_src[1] - z_src[0])
    lv0.S = S0.reshape(-1, nt)
    lv1.S = _block_mean(S0, f1)[0].reshape(-1, nt); lv2.S = _block_mean(S0, f2)[0].reshape(-1, nt)
    key = _kernel_key(rec_grid, s_src, z_src, trace, mu, nu, mode, theta, f1, f2, D.shape)
    path = os.path.join(cache_dir, f'qs_kernel_{key}.npz') if cache_dir else None
    cached = None
    if path and os.path.exists(path):
        z = np.load(path)
        cached = (z['K'], z['ptr'])
        if verbose: print(f'   qs {mode}: using cached kernel {os.path.basename(path)} ({cached[0].nbytes/1e9:.2f} GB)', flush=True)
    store, ptr = [], [0]
    tot_bytes = 0; keep = bool(path) and cached is None
    s_of, z_of = np.asarray(s_rec, float), np.asarray(z_rec, float)
    n = nsr * no * nzr; nk = len(kidx)
    peakF = np.empty(n); tpeak = np.empty(n); finF = np.empty(n); peakSJ = np.empty(n); dsm_min = np.empty(n); F0 = np.empty(n)
    Fsl = np.zeros((n, nk)); dfin = np.empty((n, 6)); dsl = np.zeros((n, nk, 6)); Fmx = np.zeros((n, nk)); Fsus = np.empty(n); Fsus_sl = np.zeros((n, nk))
    sl = np.tile([[-1.0, 0.0, 0.0]], (1, 1)); t_start = time.time(); tot_p = 0; g = 0
    for a in range(nsr):
        chunk = np.empty((no, nzr, nt, 6))
        for c in range(nzr):
            sa, zc = s_of[a], z_of[c]
            u2 = np.hypot(lv2.sc - sa, lv2.zc - zc) >= theta * lv2.size
            d1 = np.hypot(lv1.sc - sa, lv1.zc - zc) >= theta * lv1.size
            u1 = (~u2[par1_2]) & d1
            u0 = (~u2[par0_2]) & (~u1[par0_1])
            Sh = np.concatenate([lv2.S[u2], lv1.S[u1], lv0.S[u0]]).astype(np.float64)       # [P, nt]
            P = Sh.shape[0]; tot_p += P
            if cached is not None:
                Kp = cached[0][cached[1][g]:cached[1][g + 1]].reshape(no, P, 6)
            else:
                tris = np.concatenate([lv2.tris[u2], lv1.tris[u1], lv0.tris[u0]]).reshape(-1, 3, 3)
                obs = rec_grid[a, :, c, :]
                op = np.repeat(obs, tris.shape[0], axis=0); tp = np.tile(tris, (no, 1, 1))
                E = M.strain(op, tp, np.tile(sl, (len(op), 1)), nu)
                Sg = M.strain_to_stress(E, mu, nu)[:, [0, 1, 2, 3, 5, 4]]
                Kp = Sg.reshape(no, P, 2, 6).sum(2)
                if keep:
                    store.append(Kp.astype(np.float32).ravel()); ptr.append(ptr[-1] + store[-1].size)
                    tot_bytes += store[-1].nbytes
                    if tot_bytes > cache_max_gb * 1e9:
                        keep = False; store = []; ptr = [0]
                        if verbose: print(f'   kernel cache would exceed {cache_max_gb:g} GB; not caching', flush=True)
            g += 1
            chunk[:, c] = np.matmul(Kp.astype(np.float64).transpose(0, 2, 1), Sh).transpose(0, 2, 1)   # [no, nt, 6]
        sl_ = slice(a * no * nzr, (a + 1) * no * nzr)
        out = screen_from_acc(chunk.reshape(-1, nt, 6), init[sl_], coh[sl_], phi[sl_], pf[sl_], dt, kidx, nsus)
        for arr, val in zip((peakF, tpeak, finF, peakSJ, dsm_min, F0, Fsl, dfin, dsl, Fmx, Fsus, Fsus_sl), out):
            arr[sl_] = val
        if verbose and (a % max(1, nsr // 10) == 0 or a == nsr - 1):
            print(f'   qs {mode}: transect {a+1}/{nsr}, {tot_p/((a+1)*nzr):.0f} patches/receiver group, '
                  f'{time.time()-t_start:.0f} s', flush=True)
    if keep and store:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, K=np.concatenate(store), ptr=np.array(ptr, np.int64))
        if verbose: print(f'   qs {mode}: saved kernel cache {os.path.basename(path)} ({sum(x.nbytes for x in store)/1e9:.2f} GB)', flush=True)
    return peakF, tpeak, finF, peakSJ, dsm_min, F0, Fsl, dfin, dsl, Fmx, Fsus, Fsus_sl


def qs_history(rec_grid, s_rec, z_rec, trace, s_src, z_src, D, t0, T, times, mu, nu, mode='full', theta=2.5, f1=5, f2=25,
               verbose=True):
    """Quasi-static stress history acc[n_rec, nt, 6] (Pa) of all receivers.
    rec_grid [ns_r, n_off, nz_r, 3] (z negative down); s_rec [ns_r] arc length of each transect, z_rec [nz_r] depths (m, >0).
    One hierarchy selection is made per (transect, depth) group."""
    M = _F if mode == 'full' else _H
    ns, nz = D.shape
    L = ns * (s_src[1] - s_src[0]); W = nz * (z_src[1] - z_src[0])
    ds, dz = L / ns, W / nz
    nt = len(times)
    S0 = slip_snapshots(D, t0, T, times)
    lv0 = _Level(trace, ds, dz, np.arange(ns), np.arange(1, ns + 1), np.arange(nz), np.arange(1, nz + 1), S0)
    S1, i0, i1, j0, j1 = _block_mean(S0, f1); lv1 = _Level(trace, ds, dz, i0, i1, j0, j1, S1)
    S2, i0, i1, j0, j1 = _block_mean(S0, f2); lv2 = _Level(trace, ds, dz, i0, i1, j0, j1, S2)
    # parent maps (cell -> block) for the hierarchy
    nb1z = lv1.shape[1]; nb2z = lv2.shape[1]
    ci, cj = np.meshgrid(np.arange(ns), np.arange(nz), indexing='ij'); ci, cj = ci.ravel(), cj.ravel()
    par0_1 = (ci // f1) * nb1z + (cj // f1); par0_2 = (ci // f2) * nb2z + (cj // f2)
    b1i, b1j = np.meshgrid(np.arange(lv1.shape[0]), np.arange(nb1z), indexing='ij')
    par1_2 = ((b1i.ravel() * f1) // f2) * nb2z + (b1j.ravel() * f1) // f2

    nsr, no, nzr, _ = rec_grid.shape
    out = np.zeros((nsr, no, nzr, nt, 6))
    s_of, z_of = np.asarray(s_rec, float), np.asarray(z_rec, float)
    tot_p = 0; t_start = time.time(); sl = np.tile([[-1.0, 0.0, 0.0]], (1, 1))
    for a in range(nsr):
        for c in range(nzr):
            sa, zc = s_of[a], z_of[c]
            u2 = np.hypot(lv2.sc - sa, lv2.zc - zc) >= theta * lv2.size            # use level-2 block as is
            d1 = np.hypot(lv1.sc - sa, lv1.zc - zc) >= theta * lv1.size
            u1 = (~u2[par1_2]) & d1                                                  # level-1 block as is
            u0 = (~u2[par0_2]) & (~u1[par0_1])                                       # remaining cells
            tris = np.concatenate([lv2.tris[u2], lv1.tris[u1], lv0.tris[u0]]).reshape(-1, 3, 3)
            Sh = np.concatenate([lv2.S[u2], lv1.S[u1], lv0.S[u0]])                   # [P, nt]
            P = Sh.shape[0]; tot_p += P
            obs = rec_grid[a, :, c, :]                                               # [no,3]
            op = np.repeat(obs, tris.shape[0], axis=0); tp = np.tile(tris, (no, 1, 1))
            E = M.strain(op, tp, np.tile(sl, (len(op), 1)), nu)                       # unit RL slip, xx yy zz xy xz yz
            Sg = M.strain_to_stress(E, mu, nu)[:, [0, 1, 2, 3, 5, 4]]                 # -> xx yy zz xy yz xz
            Kp = Sg.reshape(no, P, 2, 6).sum(2)                                      # sum the 2 triangles of each patch
            out[a, :, c] = np.einsum('opk,pt->otk', Kp, Sh.astype(np.float64))
        if verbose and (a % max(1, nsr // 10) == 0 or a == nsr - 1):
            print(f'   qs {mode}: transect {a+1}/{nsr}, {tot_p/((a+1)*nzr):.0f} patches/receiver group, '
                  f'{time.time()-t_start:.0f} s', flush=True)
    return out.reshape(-1, nt, 6)


@nb.njit(cache=True)
def _dp1(s, coh, sphi, cphi, pf):
    sm = (s[0] + s[1] + s[2]) / 3.0
    a = s[0] - sm; b = s[1] - sm; c = s[2] - sm
    j2 = 0.5 * (a * a + b * b + c * c) + s[3] ** 2 + s[4] ** 2 + s[5] ** 2
    sj = np.sqrt(j2)
    return sj - (coh * cphi - (sm + pf) * sphi), sj, sm


@nb.njit(parallel=True, cache=True)
def screen_from_acc(acc, init, coh, phi, pf, dt, kidx, nsus):
    """same outputs as kinstress.yield_screen, from a stress-change history acc[n,nt,6]"""
    n, nt, _ = acc.shape
    peakF = np.empty(n); tpeak = np.empty(n); finF = np.empty(n); peakSJ = np.empty(n)
    dsm_min = np.empty(n); F0 = np.empty(n); Fsl = np.zeros((n, kidx.shape[0])); dfin = np.empty((n, 6))
    dsl = np.zeros((n, kidx.shape[0], 6)); Fmx = np.zeros((n, kidx.shape[0]))
    Fsus = np.empty(n); Fsus_sl = np.zeros((n, kidx.shape[0]))
    for i in nb.prange(n):
        sp_ = np.sin(phi[i]); cp_ = np.cos(phi[i])
        s = np.empty(6); Fk = np.empty(nt)
        F0[i], _, _ = _dp1(init[i], coh[i], sp_, cp_, pf[i])
        best = -1e30; tb = 0.0; bsj = 0.0; dmin = 0.0
        for k in range(nt):
            for c in range(6):
                s[c] = init[i, c] + acc[i, k, c]
            F, sj, sm = _dp1(s, coh[i], sp_, cp_, pf[i])
            Fk[k] = F
            if F > best:
                best = F; tb = k * dt; bsj = sj
            dsm = (acc[i, k, 0] + acc[i, k, 1] + acc[i, k, 2]) / 3.0
            if dsm < dmin:
                dmin = dsm
        for j in range(kidx.shape[0]):
            for c in range(6):
                s[c] = init[i, c] + acc[i, kidx[j], c]
                dsl[i, j, c] = acc[i, kidx[j], c]
            Fsl[i, j], _, _ = _dp1(s, coh[i], sp_, cp_, pf[i])
            m_ = -1e30
            for k in range(kidx[j] + 1):
                if Fk[k] > m_:
                    m_ = Fk[k]
            Fmx[i, j] = m_
        ns_ = max(1, min(nsus, nt))
        Fsk = np.empty(nt)
        for k in range(nt):
            if k < ns_ - 1:
                Fsk[k] = -1e30
            else:
                mm = Fk[k]
                for q in range(1, ns_):
                    if Fk[k - q] < mm:
                        mm = Fk[k - q]
                Fsk[k] = mm
        mx = -1e30
        for k in range(nt):
            if Fsk[k] > mx:
                mx = Fsk[k]
        Fsus[i] = mx
        for j in range(kidx.shape[0]):
            m_ = -1e30
            for k in range(kidx[j] + 1):
                if Fsk[k] > m_:
                    m_ = Fsk[k]
            Fsus_sl[i, j] = m_
        for c in range(6):
            s[c] = init[i, c] + acc[i, nt - 1, c]
            dfin[i, c] = acc[i, nt - 1, c]
        Ff, _, _ = _dp1(s, coh[i], sp_, cp_, pf[i])
        peakF[i] = best; tpeak[i] = tb; finF[i] = Ff; peakSJ[i] = bsj; dsm_min[i] = dmin
    return peakF, tpeak, finF, peakSJ, dsm_min, F0, Fsl, dfin, dsl, Fmx, Fsus, Fsus_sl
