"""
run_compare.py - Stage 1: run the SAME slip / rupture-time model with the dynamic full-space solver (Aki & Richards),
the quasi-static full-space solver and the quasi-static half-space (free surface) solver, then compare.
Spyder: edit CONFIG, run cell by cell.   Needs:  pip install cutde
"""
# %% CONFIG ---------------------------------------------------------------------------------
import os, sys
WORKDIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
os.chdir(WORKDIR); sys.path.insert(0, WORKDIR)

OUTDIR = "results"                              # <-- change to your output folder; relative paths are relative to this script's folder
NAME = 'st1'                                    # outputs: OUTDIR/st1_ar.npz ... and OUTDIR/st1_profiles.png etc.
GEO = 'Elsinore_simple30m_smooth1.geo'
SOURCE = ['--max-slip', '3', '--vr', '2700', '--rise', '1.5', '--shallow-frac', '1.0', '--hypo', 'M']
# any other source options go in SOURCE, e.g. a stochastic one:
# SOURCE = ['--slip', '2', '--slip-sigma', '0.4', '--ssd', '50', '--vr', '2700', '--vr-std', '300', '--seed', '7', '--hypo', 'M']
GRID = ['--h', '100', '--dz', '250', '--ds', '300', '--offsets', '300,400,600', '--tmax', '15', '--save-times', '4,8']   # with 'ar' keep the smallest offset >= 3*h
SOLVERS = ['ar', 'qs-full', 'qs-half']          # first one is the reference in the comparison
TRANSECTS = '31,62,95'                          # cross-sections to draw (1-based)

# %% RUN each solver on identical input --------------------------------------------------------
import elsinore_yield as ey
OUTDIR = os.path.abspath(OUTDIR) if os.path.isabs(OUTDIR) else os.path.join(WORKDIR, OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)
NAME = os.path.join(OUTDIR, NAME)
for s in SOLVERS:
    if os.path.exists(f'{NAME}_{s}.npz'):
        print(f'{NAME}_{s}.npz exists, skipping'); continue
    print(f'\n===== solver {s} =====')
    ey.main(['--geo', GEO, *SOURCE, *GRID, '--solver', s, '--out', f'{NAME}_{s}'])

# %% COMPARE: table, depth profiles, scatter, per-transect panels --------------------------------
import matplotlib.pyplot as plt, compare_runs as cr
cr.main([f'{s}={NAME}_{s}.npz' for s in SOLVERS] + ['--out', NAME, '--transects', TRANSECTS])
plt.show()
print(open(f'{NAME}_table.txt').read())
