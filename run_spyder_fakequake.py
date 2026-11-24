#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:41:33 2026

@author: emarschall
"""

"""
run_spyder.py - run ONE model and save the figures.   (Spyder: edit CONFIG, then run cell by cell with Ctrl+Enter,
or the whole file with F5.  Terminal:  python run_spyder.py)

Everything is written to  OUTDIR/<NAME>*  (set OUTDIR below):
    <NAME>.npz                   all numbers (open in Python with np.load)
    <NAME>.vtk                   receivers + yield fields for ParaView
    <NAME>_slip_rt.png           slip with rupture-time contours (0 km depth at top, SE on the right)
    <NAME>_sections_final/T###.png   one yielding cross-section per transect (map of the fault with a dot for the section)
    <NAME>_slices/tNN.NNs/       for every time in SAVE_TIMES: yield, normal-stress-% and mean-stress-% cross-sections
                                 and the plan-view fault figure (slip + rupture time)
    <NAME>_fault3d.png, _fault.vtk   3-D view of the fault
"""
# %% CONFIG ---------------------------------------------------------------------------------
import os, sys
import numpy as np
# os.environ['NUMBA_NUM_THREADS'] = '8'; os.environ['OMP_NUM_THREADS'] = '8'   # optional: by default all cores are used; set your core count (or fewer) BEFORE anything else runs (restart the kernel after changing)

WORKDIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
os.chdir(WORKDIR); sys.path.insert(0, WORKDIR)


# ---- WHERE THE OUTPUT GOES -------------------------------------------------------------------------
OUTDIR = "results_het"       # <-- your output folder, e.g. "/Users/me/elsinore/run1" (Mac) or r"C:\Users\me\run1" (Windows).
                         #     Relative paths are relative to this script's folder.  Created if it does not exist.
NAME   = 'example_het'     # file-name prefix: OUTDIR/example.npz, OUTDIR/example_slices/ ...
GEO    = 'Elsinore_simple30m_smooth1.geo'

# ---- PHYSICS --------------------------------------------------------------------------------------
SOLVER = 'qs-half'     # 'ar'       dynamic, full space (Aki & Richards).  Keep smallest offset >= 3*H.
                       # 'qs-full'  quasi-static dislocation patches, full space
                       # 'qs-half'  quasi-static dislocation patches, half-space WITH FREE SURFACE (needs: pip install cutde)

# ---- SOURCE: slip, rupture time, rise time --------------------------------------------------------
FIELD_SLIP = None      # 'lat_lon-data.txt' -> use field-measured slip instead of the synthetic source below
SEISSOL_FAULT = None   # '/path/to/<prefix>-fault.xdmf' -> use a SeisSol fault output as the source (final slip ASl, rupture time RT, peak slip rate PSR -> rise time;
                       #   needs: pip install seissolxdmf h5py).  Overrides the synthetic source below (MAX_SLIP, VR, RISE ... are then ignored); FIELD_SLIP must be None.
                       #   Use the SAME .geo as the SeisSol mesh, and STRESS_SOURCE = 'yaml' with your SeisSol yaml below so the initial stress matches.
MAX_SLIP   = None      # m, peak slip.  Set to None and give MEAN_SLIP to pin the mean instead
MEAN_SLIP  = 1.5
VR, RISE   = 2700.0, 1.5               # rupture speed (m/s), rise time (s)
RISE_MODE = 'constant' # 'constant' = RISE (above) everywhere;  'gp' = Graves & Pitarka (2016): rise time ~ sqrt(local slip), x2 above 5 km depth (linear to x1 at 8 km),
                       #   fault average set from the moment: 1.6e-9 * M0^(1/3) (M0 in dyne-cm), so it grows with magnitude.  'gp' ignores RISE.
RISE_MEAN = None       # gp: pin the fault-average rise time (s) instead of the moment scaling, e.g. 1.0.  None = self-similar scaling
RISE_ETA = 0.8         # gp: correlation of rise time with slip.  1 = rise ~ sqrt(slip) exactly, 0 = unrelated, 0.8 = the value used by Graves & Pitarka (2016)
RISE_DEPTH_FACTOR = True   # gp: double the rise time in the shallow zone (0-5 km, transition to 1x at 8 km).  False = no depth dependence
HYPO, HYPO_DEPTH = 'N', 8000.0         # 'N', 'M', 'S' or 'x,y' (local m)
SHALLOW_FRAC = 1.0     # slip at the surface relative to depth >= 3 km (1 = none).  Use SSD_PCT below for a shallow-slip deficit box
TAPER_S = 0       # m: slip tapers to zero over this length at each END of the fault (along strike).  0 = no taper: full slip up to the fault ends
TAPER_Z = 0       # m: slip tapers to zero over this length above the fault BASE (15 km).  0 = no taper: full slip down to the base.  (The surface is never tapered.)
# ... optional stochastic extras (all off by default) .............................................
SLIP_SIGMA = 0.4       # >0: smooth random slip, log-normal std (0.3 mild ... 0.6 strong)
CORR_S, CORR_Z = 2000.0, 2000.0        # correlation lengths along strike / down dip (m)
SSD_PCT, SSD_DEPTH = 100, 1000.0      # e.g. 50 -> slip at the surface is 50 % of the mean slip below SSD_DEPTH
SSD_SMOOTH = 0.0                       # 0 = abrupt step at SSD_DEPTH ... 1 = slip rises smoothly over the whole 0..SSD_DEPTH layer
VR_STD, VR_SEG = 400.0, 1000.0           # >0: rupture speed sampled every VR_SEG m from N(VR, VR_STD)
VR_SEG_Z = 0.0                         # >0 also samples with depth
VR_MIN, VR_MAX = 1000.0, 3300.0        # clip (m/s); S-wave speed is 3464
SEED = 1
# ... field-slip runs: scale the slip with depth, optionally add a random component ...............
DEPTH_SCALE = 1.0                      # slip below DEPTH_SCALE_DEPTH = DEPTH_SCALE x the surface (field) slip  (e.g. 1.5)
DEPTH_SCALE_DEPTH = 1000.0             # m
DEPTH_SCALE_SMOOTH = 1.0               # 0 = abrupt step at DEPTH_SCALE_DEPTH ... 1 = gradual over the whole 0..DEPTH_SCALE_DEPTH layer
FIELD_NOISE = 0.0                      # >0: smooth random multiplier on the field slip (log-normal std; uses CORR_S, CORR_Z, SEED)
FIELD_NOISE_DEPTH = 3000.0             # m: the random part grows from 0 at the surface to full here (0 = all depths)

# ---- RESOLUTION / RECEIVERS -----------------------------------------------------------------------
H, DZ, DS = 100.0, 500.0, 300.0        # source cell (m), receiver depth spacing (m), spacing along strike (m)
OFFSETS = [25,50,100,150,250,500,1000,2000]     # m from the fault, both sides (dashed lines on the plots mark 1.5h and 3h)
TMAX, DT = 16.0, 0.25                  # s: simulated time, time step.  DT <= (shortest rise time)/6 gives converged peak F (rise 1.5 s -> 0.25 s is enough; 0.1 s also resolves Tv = 0.3 s); DT = 2 s underestimates it by 2-3x
TRANSECTS = "/Users/emarschall/Desktop/Off_fault_Tom_Ashley/receivers_transects_perp_smooth1.dat"                        # 'receivers_transects_perp_smooth1.dat' -> your own transects (None = one every DS m)
TRANSECT_SUBSET = '4,6,8,12,16,20,24,28,32,36,38,42,48'                 # use only some of the transects in the file (or the DS-spaced ones):
                                       #   'all' | 'every:5' (1, 6, 11, ...) | 'n:10' (10 evenly spaced) | '1,6,11,21' (numbers in the file, 1-based)
                                       #   The figures then count the chosen transects 1, 2, 3 ... (the run prints the original numbers).

# ---- TIME SLICES: times (s after rupture starts) at which you want pictures -------------------------
SAVE_TIMES = [2,4,6,8]                # each time is rounded to the nearest time step DT.  [] = no slice figures.
                                       # For every time: yield cross-sections, normal-stress-% cross-sections (+ mean-stress-% if listed
                                       # below) and a plan view of slip + rupture time on the fault.
SLICE_SECTIONS = 'all'                 # which transects get slice figures: 'all' or e.g. '10,31,62,95' (1-based).  'all' = many PNGs
SLICE_STRESS   = ['mean', 'j2']    # 'normal' = fault-normal stress change in % of initial (+ = clamping); 'mean' = mean-stress change in %;
                                       # 'j2' = change of sqrt(J2) (deviatoric stress intensity) in % of its initial value (+ = more deviatoric stress)
SLICE_YIELD   = 'max'                 # yield figures: 'max' = everything that has yielded so far (max of F up to t; yielding cannot reverse),
                                       # 'instant' = F at the instant t only (can fall back below 0), 'both' = both sets of PNGs
SLICE_DPI      = 130                   # lower (100) = faster / smaller files

# ---- YIELD CRITERION ------------------------------------------------------------------------------
COH, BULK_FRICTION = 5e6, 0.65         # Pa, tan(phi)  (bulk Drucker-Prager; NOT the fault cohesion in the SeisSol yaml)

# ---- WHICH F OUTLINES TO DRAW, and the SeisSol viscoplastic time ----------------------------------
F_LEVELS = [0]                  # MPa: one colored outline per level on the yield sections (F = 0 is the yield surface itself).  [0] = only F = 0
TV = 0                               # s, SeisSol viscoplastic relaxation time Tv.  Adds an outline (pink) around the points where F > 0 lasted for at least
                                       # ceil(TV/DT) consecutive time steps ("sustained yield").  Needs DT < TV to resolve it (DT = 0.25 s: 2 steps).  0 = off
GUIDE_LINES = False                    # draw the dashed 1.5h / 3h accuracy lines on the cross-sections (outlines are dashed inside 1.5h).  False = clean figures

# ---- INITIAL (BACKGROUND) STRESS: choose ONE of the two sources -------------------------------------
STRESS_SOURCE = 'yaml'              # 'formula' = defined by the numbers just below;  'yaml' = read from your SeisSol yaml
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
SECTIONS  = 'all'      # final-state cross-sections: 'all' or e.g. '10,31,62,95' (transect numbers, 1-based)
SLIP_MAX  = None       # fix the slip color-scale top (m) to compare runs, None = automatic
STRESS_DROP_SOLVER = None              # solver for the stress-drop summary: None = the same as SOLVER; or 'qs-full' / 'qs-half' / 'ar'
EXTRA = []             # any other elsinore_yield.py option, e.g. ['--shmax', '350']

# %% RUN the solver ---------------------------------------------------------------------------
OUTDIR = os.path.abspath(OUTDIR) if os.path.isabs(OUTDIR) else os.path.join(WORKDIR, OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)
import elsinore_yield as ey
OUT = os.path.join(OUTDIR, NAME)
print('output folder:', OUTDIR)
argv = ['--geo', GEO, '--solver', SOLVER, '--h', str(H), '--dz', str(DZ), '--ds', str(DS),
        '--offsets', ','.join(map(str, OFFSETS)), '--dt', str(DT), '--tmax', str(TMAX),
        '--hypo', str(HYPO), '--hypo-depth', str(HYPO_DEPTH), '--coh', str(COH), '--bulk-friction', str(BULK_FRICTION), '--tv', str(TV),
        '--seed', str(SEED), '--qs-theta', str(QS_THETA), '--kernel-cache-max-gb', str(KERNEL_CACHE_MAX_GB), '--out', OUT]
if not KERNEL_CACHE: argv += ['--no-kernel-cache']
if FIELD_SLIP:
    argv += ['--field-slip', FIELD_SLIP, '--rise', str(RISE), '--vr', str(VR),
             '--depth-scale', str(DEPTH_SCALE), '--depth-scale-depth', str(DEPTH_SCALE_DEPTH), '--depth-scale-smooth', str(DEPTH_SCALE_SMOOTH),
             '--field-noise', str(FIELD_NOISE), '--field-noise-depth', str(FIELD_NOISE_DEPTH), '--corr-s', str(CORR_S), '--corr-z', str(CORR_Z)]
else:
    argv += ['--rise', str(RISE), '--vr', str(VR), '--shallow-frac', str(SHALLOW_FRAC), '--slip', str(MEAN_SLIP),
             '--slip-sigma', str(SLIP_SIGMA), '--corr-s', str(CORR_S), '--corr-z', str(CORR_Z),
             '--vr-std', str(VR_STD), '--vr-seg', str(VR_SEG), '--vr-seg-z', str(VR_SEG_Z),
             '--vr-min', str(VR_MIN), '--vr-max', str(VR_MAX)]
    if MAX_SLIP: argv += ['--max-slip', str(MAX_SLIP)]
    if SSD_PCT is not None: argv += ['--ssd', str(SSD_PCT), '--ssd-depth', str(SSD_DEPTH), '--ssd-smooth', str(SSD_SMOOTH)]
if SEISSOL_FAULT: argv += ['--seissol-fault', SEISSOL_FAULT]
argv += ['--taper-s', str(TAPER_S), '--taper-z', str(TAPER_Z), '--rise-mode', RISE_MODE, '--rise-eta', str(RISE_ETA)] + (['--rise-mean', str(RISE_MEAN)] if RISE_MEAN else []) + ([] if RISE_DEPTH_FACTOR else ['--no-rise-depth-factor'])
if TRANSECTS: argv += ['--transects', TRANSECTS]
argv += ['--transect-subset', str(TRANSECT_SUBSET)]
if SAVE_TIMES: argv += ['--save-times', ','.join(map(str, SAVE_TIMES))]
if STRESS_SOURCE == 'yaml':
    argv += ['--seissol-yaml', SEISSOL_YAML]
elif STRESS_SOURCE == 'formula':
    argv += ['--litho-coef', str(LITHO_COEF), '--clamp-depth', str(CLAMP_DEPTH), '--shear-ratio', str(SHEAR_RATIO), '--shmax', str(SHMAX)]
    if LAYERS: argv += ['--layers', LAYERS]
else:
    raise ValueError("STRESS_SOURCE must be 'formula' or 'yaml'")
ey.main(argv + EXTRA)

# %% FIGURES: slip + rupture-time contours, and one PNG per yielding cross-section --------------
import matplotlib.pyplot as plt, plot_sections as ps
args = [OUT + '.npz', '--levels', ','.join(str(v) for v in F_LEVELS)] + ([] if GUIDE_LINES else ['--no-guides'])
if SECTIONS != 'all': args += ['--transects', SECTIONS]
ps.main(args)                       # final-state sections.  Other views: add '--what','peak'  or  '--what','time','--time','8'
ps.main(args + ['--what', 'peak', '--no-slip-fig'])   # peak over time: everything that yielded at any time (maximum of F; yielding cannot reverse)
for w in ('dmean', 'dmeanpct', 'dj2pct'):     # final mean-stress change (MPa, % of initial) and sqrt(J2) change (% of initial), with the F outlines, same transects
    ps.main(args + ['--what', w, '--no-slip-fig'])
d = np.load(OUT + '.npz')
if SLIP_MAX: ps.plot_slip_rt(d, OUT + '_slip_rt.png', slip_max=SLIP_MAX)
ps.plot_depth_profile(d, OUT + '_slip_depth_profile.png')      # slip vs depth, with the shallow transition layer marked

# %% TIME-SLICE FIGURES (for every time in SAVE_TIMES) --------------------------------------------
if SAVE_TIMES:
    import plot_slices as psl
    psl.make_slices(OUT + '.npz', transects=SLICE_SECTIONS, stress=tuple(SLICE_STRESS), dpi=SLICE_DPI, yield_mode=SLICE_YIELD, slip_max=SLIP_MAX, levels=F_LEVELS, guides=GUIDE_LINES)

# %% show the slip figure here ---------------------------------------------------------------------
img = plt.imread(OUT + '_slip_rt.png'); plt.figure(figsize=(11, 5.5)); plt.imshow(img); plt.axis('off'); plt.show()

# %% 3-D VIEW of the fault (type  %matplotlib qt  in the console first to rotate it) ----------------
import plot_fault3d as pf
pf.plot_fault_3d(d, out=OUT + '_fault3d.png', show=True, slip_max=SLIP_MAX)
pf.write_fault_vtk(d, OUT + '_fault.vtk')

# %% STRESS DROP summary (area-mean, slip-weighted, Eshelby, core region) ---------------------------
import subprocess
subprocess.run([sys.executable, 'stress_drop.py', OUT + '.npz', '--geo', GEO, '--solver', STRESS_DROP_SOLVER or SOLVER], check=False)

# %% quick numbers -------------------------------------------------------------------------------
print('final F>0 at %.1f %% of receivers;  peak F>0 at %.1f %%;  max peak F = %.1f MPa'
      % (100 * (d['finalF'] > 0).mean(), 100 * (d['peakF'] > 0).mean(), d['peakF'].max() / 1e6))
