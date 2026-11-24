"""
run_suite.py - a suite of stochastic kinematic models (random slip, random rupture speed, shallow slip deficit)
and the yielding statistics over the suite.   Spyder: edit CONFIG, run cell by cell.  Terminal: python run_suite.py

Outputs (in OUTDIR/NAME/):  run_000.npz ...   one npz per realization (same format as elsinore_yield.py)
                      suite_summary.npz  P(yield, final), P(yield, peak), mean/std of final F, per-run table
                      suite_table.csv    one line per realization (Mw, mean/max slip, speed, yielding fraction)
                      sections_Pfinal/T###.png   probability-of-yielding cross-sections (only if PLOT_PROB = True)
                      run_###_slip_rt.png, run_###_fault3d.png   slip + rupture-time contours, 2-D and 3-D, per realization
"""
# %% CONFIG ---------------------------------------------------------------------------------
import os, sys, csv, time
import numpy as np
WORKDIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
os.chdir(WORKDIR); sys.path.insert(0, WORKDIR)

# ---- WHERE THE OUTPUT GOES -------------------------------------------------------------------------
OUTDIR = "results"       # <-- your output folder, e.g. "/Users/me/elsinore/run1" (Mac) or r"C:\Users\me\run1" (Windows).
                         #     Relative paths are relative to this script's folder.  Created if it does not exist.
NAME   = 'suite_A'       # sub-folder for this suite: everything goes to OUTDIR/suite_A/
GEO    = 'Elsinore_simple30m_smooth1.geo'

# ---- THE SUITE ------------------------------------------------------------------------------------
N_RUNS = 10                            # number of realizations
SEED0  = 100                           # realization k uses seed SEED0 + k
HYPOS  = ['M']                         # cycled over the realizations, e.g. ['N','M','S']; or 'x,y' strings (local m)

# ---- PHYSICS --------------------------------------------------------------------------------------
SOLVER = 'qs-half'     # 'ar'       dynamic, full space (Aki & Richards).  Keep smallest offset >= 3*H.
                       # 'qs-full'  quasi-static dislocation patches, full space
                       # 'qs-half'  quasi-static dislocation patches, half-space WITH FREE SURFACE (needs: pip install cutde)

# ---- SOURCE: slip, rupture time, rise time (stochastic, smooth) -----------------------------------
MAX_SLIP   = 4.0       # m, peak slip.  Set to None and give MEAN_SLIP to pin the mean instead
MEAN_SLIP  = 2.0
VR, RISE   = 2700.0, 1.5               # rupture speed (m/s), rise time (s)
RISE_MODE = 'constant' # 'constant' = RISE (above) everywhere;  'gp' = Graves & Pitarka (2016): rise time ~ sqrt(local slip), x2 above 5 km depth (linear to x1 at 8 km),
                       #   fault average set from the moment: 1.6e-9 * M0^(1/3) (M0 in dyne-cm), so it grows with magnitude.  'gp' ignores RISE.
RISE_MEAN = None       # gp: pin the fault-average rise time (s) instead of the moment scaling, e.g. 1.0.  None = self-similar scaling
RISE_ETA = 0.8         # gp: correlation of rise time with slip.  1 = rise ~ sqrt(slip) exactly, 0 = unrelated, 0.8 = the value used by Graves & Pitarka (2016)
RISE_DEPTH_FACTOR = True   # gp: double the rise time in the shallow zone (0-5 km, transition to 1x at 8 km).  False = no depth dependence
HYPO_DEPTH = 8000.0                    # m (the hypocenter location itself comes from HYPOS)
SHALLOW_FRAC = 1.0     # slip at the surface relative to depth >= 3 km (1 = none).  Use SSD_PCT below for a shallow-slip deficit box
TAPER_S = 3000.0       # m: slip tapers to zero over this length at each END of the fault (along strike).  0 = no taper: full slip up to the fault ends
TAPER_Z = 4000.0       # m: slip tapers to zero over this length above the fault BASE (15 km).  0 = no taper: full slip down to the base.  (The surface is never tapered.)
# ... stochastic extras ...........................................................................
SLIP_SIGMA = 0.4       # >0: smooth random slip, log-normal std (0.3 mild ... 0.6 strong)
CORR_S, CORR_Z = 2000.0, 2000.0        # correlation lengths along strike / down dip (m)
SSD_PCT, SSD_DEPTH = 100, 1000.0      # e.g. 50 -> slip at the surface is 50 % of the mean slip below SSD_DEPTH
SSD_SMOOTH = 1.0                       # 0 = abrupt step at SSD_DEPTH ... 1 = slip rises smoothly over the whole 0..SSD_DEPTH layer
VR_STD, VR_SEG = 0.0, 1000.0           # >0: rupture speed sampled every VR_SEG m from N(VR, VR_STD)
VR_SEG_Z = 0.0                         # >0 also samples with depth
VR_MIN, VR_MAX = 1000.0, 3300.0        # clip (m/s); S-wave speed is 3464

# ---- RESOLUTION / RECEIVERS -----------------------------------------------------------------------
H, DZ, DS = 100.0, 500.0, 300.0        # source cell (m), receiver depth spacing (m), spacing along strike (m)
OFFSETS = [0.01, 1, 5, 25,50,100,150,200,400, 600,800,1000]     # m from the fault, both sides (dashed lines on the plots mark 1.5h and 3h)
TMAX, DT = 16.0, 0.1                  # s: simulated time, time step.  DT <= (shortest rise time)/6 gives converged peak F (rise 1.5 s -> 0.25 s is enough; 0.1 s also resolves Tv = 0.3 s); DT = 2 s underestimates it by 2-3x
TRANSECTS = "/Users/emarschall/Desktop/Off_fault_Tom_Ashley/receivers_transects_perp_smooth1.dat"                        # 'receivers_transects_perp_smooth1.dat' -> your own transects (None = one every DS m)
TRANSECT_SUBSET = '4,6,8,12,16,20,24,28,32,36,38,42,48'                # use only some of the transects in the file (or the DS-spaced ones):
                                       #   'all' | 'every:5' (1, 6, 11, ...) | 'n:10' (10 evenly spaced) | '1,6,11,21' (numbers in the file, 1-based)
                                       #   The figures then count the chosen transects 1, 2, 3 ... (the run prints the original numbers).

# ---- TIME SLICES: times (s after rupture starts) at which you want pictures -------------------------
SAVE_TIMES = [2,4, 6, 8,]                # each time is rounded to the nearest time step DT.  [] = no slice figures.
                                       # For every time: yield cross-sections, normal-stress-% cross-sections (+ mean-stress-% if listed
                                       # below) and a plan view of slip + rupture time on the fault.
SLICE_SECTIONS = 'all'                 # which transects get slice figures: 'all' or e.g. '10,31,62,95' (1-based).  'all' = many PNGs
SLICE_STRESS   = ['mean']    # 'normal' = fault-normal stress change in % of initial (+ = clamping); 'mean' = mean-stress change in %
SLICE_YIELD   = 'max'                 # yield figures: 'max' = everything that has yielded so far (max of F up to t; yielding cannot reverse),
                                       # 'instant' = F at the instant t only (can fall back below 0), 'both' = both sets of PNGs
SLICE_DPI      = 130                   # lower (100) = faster / smaller files

# ---- YIELD CRITERION ------------------------------------------------------------------------------
COH, BULK_FRICTION = 5e6, 0.65         # Pa, tan(phi)  (bulk Drucker-Prager; NOT the fault cohesion in the SeisSol yaml)

# ---- WHICH F OUTLINES TO DRAW, and the SeisSol viscoplastic time ----------------------------------
F_LEVELS = [0, 5, 20]                  # MPa: one colored outline per level on the yield sections (F = 0 is the yield surface itself).  [0] = only F = 0
TV = 0.3                               # s, SeisSol viscoplastic relaxation time Tv.  Adds an outline (pink) around the points where F > 0 lasted for at least
                                       # ceil(TV/DT) consecutive time steps ("sustained yield").  Needs DT < TV to resolve it (DT = 0.25 s: 2 steps).  0 = off
GUIDE_LINES = True                    # draw the dashed 1.5h / 3h accuracy lines on the cross-sections (outlines are dashed inside 1.5h).  False = clean figures

# ---- INITIAL (BACKGROUND) STRESS: choose ONE of the two sources -------------------------------------
STRESS_SOURCE = 'formula'              # 'formula' = defined by the numbers just below;  'yaml' = read from your SeisSol yaml
# (a) 'formula':  mean stress = LITHO_COEF x (rho g integrated over depth), constant below CLAMP_DEPTH;
#                 deviatoric stress = SHEAR_RATIO x mean stress, oriented with SHmax at azimuth SHMAX
LITHO_COEF  = 0.25                     # 0.25 = "0.25 g":  mean stress = 0.25 x lithostatic pressure
CLAMP_DEPTH = 5000.0                   # m, below this depth the stress stops increasing (the SeisSol-style clamp)
SHEAR_RATIO = 13.4347 / 36.7875        # deviatoric / mean stress (0.3652 = values of your SeisSol model).  0 = isotropic
SHMAX       = 350.0                    # degrees clockwise from north, direction of maximum horizontal compression
LAYERS      = None                     # None = built-in density layers;  or '0:2300,1000:2500,3000:2600,6000:2700,10000:2750' (top depth m : density kg/m3)
# (b) 'yaml':     take the whole stress state (depth dependence, orientation) from the SeisSol yaml (needs: pip install lupa)
SEISSOL_YAML = 'fault_tensor_depthdep_TW_S.yaml'

# ---- SPEED ----------------------------------------------------------------------------------------
KERNEL_CACHE = True                    # quasi-static solvers: the first run on a given geometry saves its stress kernel (a few hundred MB,
                                       # in OUTDIR/_kernel_cache); later runs with the SAME geometry (new slip, rupture speed, rise time,
                                       # seed, stress, yield parameters ...) reuse it and take seconds.  Changing H, DZ, DS, OFFSETS,
                                       # TRANSECTS, TRANSECT_SUBSET, GEO, SOLVER or QS_THETA creates a new kernel.
KERNEL_CACHE_MAX_GB = 3.0              # do not cache if the kernel would be larger than this
QS_THETA = 2.5                         # quasi-static accuracy/speed knob: smaller = faster and less accurate (see MANUAL)

# ---- FIGURES --------------------------------------------------------------------------------------
PLOT_PROB = False      # True = draw the probability-of-yielding cross-sections (sections_Pfinal/); False = skip them
SECTIONS  = 'all'      # transects for those probability sections: 'all' or e.g. '10,31,62,95' (transect numbers, 1-based)
SLIP_MAX  = None       # fix the slip color-scale top (m) for ALL slip figures, 2-D and 3-D; None = each run uses its own peak
PLOT_3D, FAULT_VTK = True, False       # 3-D fault figure per realization; also a fault-surface .vtk for ParaView
CONTOUR_DT, ELEV, AZIM, VEXAG = 1.0, 18, None, 1.0   # rupture-time contour interval (s), camera elevation / azimuth (None = view from SW), vertical exaggeration
EXTRA = []             # any other elsinore_yield.py option, e.g. ['--shmax', '350']

# %% RUN THE SUITE -----------------------------------------------------------------------------
import elsinore_yield as ey
OUTDIR = os.path.abspath(OUTDIR) if os.path.isabs(OUTDIR) else os.path.join(WORKDIR, OUTDIR)
SUITE = os.path.join(OUTDIR, NAME)
os.makedirs(SUITE, exist_ok=True)
print('output folder:', SUITE)
rows = []; t_all = time.time()
for k in range(N_RUNS):
    out = f'{SUITE}/run_{k:03d}'
    if os.path.exists(out + '.npz'):
        print(f'[{k+1}/{N_RUNS}] {out}.npz exists, skipping'); continue
    hypo = HYPOS[k % len(HYPOS)]
    argv = ['--geo', GEO, '--solver', SOLVER, '--h', str(H), '--dz', str(DZ), '--ds', str(DS),
            '--offsets', ','.join(map(str, OFFSETS)), '--dt', str(DT), '--tmax', str(TMAX),
            '--hypo', hypo, '--hypo-depth', str(HYPO_DEPTH), '--coh', str(COH), '--bulk-friction', str(BULK_FRICTION), '--tv', str(TV),
            '--seed', str(SEED0 + k), '--qs-theta', str(QS_THETA), '--kernel-cache-max-gb', str(KERNEL_CACHE_MAX_GB), '--out', out,
            '--rise', str(RISE), '--vr', str(VR), '--shallow-frac', str(SHALLOW_FRAC), '--slip', str(MEAN_SLIP),
            '--slip-sigma', str(SLIP_SIGMA), '--corr-s', str(CORR_S), '--corr-z', str(CORR_Z),
            '--vr-std', str(VR_STD), '--vr-seg', str(VR_SEG), '--vr-seg-z', str(VR_SEG_Z),
            '--vr-min', str(VR_MIN), '--vr-max', str(VR_MAX)]
    if MAX_SLIP: argv += ['--max-slip', str(MAX_SLIP)]
    if SSD_PCT is not None: argv += ['--ssd', str(SSD_PCT), '--ssd-depth', str(SSD_DEPTH), '--ssd-smooth', str(SSD_SMOOTH)]
    argv += ['--taper-s', str(TAPER_S), '--taper-z', str(TAPER_Z), '--rise-mode', RISE_MODE, '--rise-eta', str(RISE_ETA)] + (['--rise-mean', str(RISE_MEAN)] if RISE_MEAN else []) + ([] if RISE_DEPTH_FACTOR else ['--no-rise-depth-factor'])
    if TRANSECTS: argv += ['--transects', TRANSECTS]
    argv += ['--transect-subset', str(TRANSECT_SUBSET)]
    if not KERNEL_CACHE: argv += ['--no-kernel-cache']
    if SAVE_TIMES: argv += ['--save-times', ','.join(map(str, SAVE_TIMES))]
    if STRESS_SOURCE == 'yaml':
        argv += ['--seissol-yaml', SEISSOL_YAML]
    elif STRESS_SOURCE == 'formula':
        argv += ['--litho-coef', str(LITHO_COEF), '--clamp-depth', str(CLAMP_DEPTH), '--shear-ratio', str(SHEAR_RATIO), '--shmax', str(SHMAX)]
        if LAYERS: argv += ['--layers', LAYERS]
    else:
        raise ValueError("STRESS_SOURCE must be 'formula' or 'yaml'")
    print(f'\n===== realization {k+1}/{N_RUNS}  seed {SEED0+k}  hypocenter {hypo} =====')
    ey.main(argv + EXTRA)
    if SAVE_TIMES:
        import plot_slices as psl
        psl.make_slices(out + '.npz', transects=SLICE_SECTIONS, stress=tuple(SLICE_STRESS), dpi=SLICE_DPI, yield_mode=SLICE_YIELD, verbose=False, levels=F_LEVELS, guides=GUIDE_LINES)

import plot_sections as _ps0
ps_get_h = _ps0.get_h
# %% STATISTICS --------------------------------------------------------------------------------
files = sorted(f for f in os.listdir(SUITE) if f.startswith('run_') and f.endswith('.npz'))
n = 0; cnt_fin = cnt_pk = sF = sF2 = None; rows = []
for f in files:
    d = np.load(f'{SUITE}/{f}')
    Ff, Fp = d['finalF'] / 1e6, d['peakF'] / 1e6
    if n == 0:
        cnt_fin = np.zeros_like(Ff); cnt_pk = np.zeros_like(Ff); sF = np.zeros_like(Ff); sF2 = np.zeros_like(Ff); ref = d
    cnt_fin += Ff > 0; cnt_pk += Fp > 0; sF += Ff; sF2 += Ff**2; n += 1
    D, rt = d['slip'], d['rt']; M0 = 32.4e9 * (float(d['L']) / D.shape[0]) * (float(d['W']) / D.shape[1]) * D.sum()
    rows.append(dict(run=f[:-4], hypo_s_km=float(d['hypo_s']) / 1e3, Mw=2 / 3 * (np.log10(M0) - 9.1), mean_slip=D.mean(), max_slip=D.max(),
                     duration_s=rt[D > 0.05 * D.max()].max(), frac_yield_final=(Ff > 0).mean(), frac_yield_peak=(Fp > 0).mean()))
Pfin, Ppk, mF = cnt_fin / n, cnt_pk / n, sF / n
sdF = np.sqrt(np.maximum(sF2 / n - mF**2, 0))
np.savez_compressed(f'{SUITE}/suite_summary.npz', P_final=Pfin, P_peak=Ppk, mean_finalF=mF, std_finalF=sdF, n=n, h=ps_get_h(ref),
                    s_rec=ref['s_rec'], offsets=ref['offsets'], depth_rec=ref['depth_rec'], rec=ref['rec'], trace=ref['trace'], L=ref['L'], W=ref['W'])
with open(f'{SUITE}/suite_table.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'{n} realizations: Mw {np.mean([r["Mw"] for r in rows]):.2f} +- {np.std([r["Mw"] for r in rows]):.2f}; '
      f'cells with P(final yield)>0.5: {100*(Pfin>0.5).mean():.1f}%; table in {SUITE}/suite_table.csv')

# %% PROBABILITY CROSS-SECTIONS (one PNG each) ---------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
import plot_sections as ps
d = np.load(f'{SUITE}/suite_summary.npz')
off, z, s = d['offsets'], d['depth_rec'] / 1e3, d['s_rec'] / 1e3
oo = np.arange(off.min(), off.max() + 1, 10.0); omin = np.abs(off).min()
Pr = interp1d(off, d['P_final'], axis=1)(oo)
if PLOT_PROB:
    ids = range(Pr.shape[0]) if SECTIONS == 'all' else [int(v) - 1 for v in SECTIONS.split(',')]
    os.makedirs(f'{SUITE}/sections_Pfinal', exist_ok=True)
    for i in ids:
        ps.single_section(d, i, oo, z, Pr, s, Normalize(0, 1), omin, f'P(final yield), {int(d["n"])} runs',
                          out=f'{SUITE}/sections_Pfinal/T{i+1:03d}.png', cblabel='probability of F>0;  50% outlined', level=0.5)
    print('wrote', f'{SUITE}/sections_Pfinal/')
else:
    print('probability cross-sections skipped (PLOT_PROB = False)')

# %% SLIP + RUPTURE TIME FOR EVERY REALIZATION: 2-D figure and 3-D fault view ---------------------
import plot_fault3d as pf
for f in files:
    dd = np.load(f'{SUITE}/{f}'); b = f'{SUITE}/{f[:-4]}'
    ps.plot_slip_rt(dd, b + '_slip_rt.png', slip_max=SLIP_MAX)
    ps.plot_depth_profile(dd, b + '_slip_depth_profile.png')
    if PLOT_3D:
        pf.plot_fault_3d(dd, out=b + '_fault3d.png', contour_dt=CONTOUR_DT, elev=ELEV, azim=AZIM, vexag=VEXAG, slip_max=SLIP_MAX)
        if FAULT_VTK: pf.write_fault_vtk(dd, b + '_fault.vtk')
print('wrote slip/rupture-time figures' + (' and 3-D fault views' if PLOT_3D else ''))
