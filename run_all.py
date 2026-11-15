"""Runs the whole study in order. Safe to restart: steps whose output already exists are skipped.
Usage:  python run_all.py            (everything)
        python run_all.py static     (only the tasks on the static Android data, then the summary)"""
import subprocess, sys, os, glob, time, config
env = dict(os.environ, OMP_NUM_THREADS=str(config.N_JOBS), OPENBLAS_NUM_THREADS=str(config.N_JOBS), MKL_NUM_THREADS=str(config.N_JOBS))
W = config.W; here = os.path.dirname(os.path.abspath(__file__))
def step(name, args, done):
    if done and all(os.path.exists(W + d) for d in done): print('skip', name); return
    t = time.time(); print('>>>', name, flush=True)
    r = subprocess.run([sys.executable] + args, cwd=here, env=env)
    if r.returncode: sys.exit(f'FAILED: {name}')
    print(f'<<< {name} finished in {(time.time() - t) / 60:.1f} min', flush=True)
TASKS = [('andmal_static', 'bin'), ('andmal_static', 'multi'), ('maldroid', 'bin'), ('maldroid', 'multi'), ('apt_flow', 'bin'), ('unraveled', 'bin'), ('andmal_dyn', 'multi')]
if len(sys.argv) > 1 and sys.argv[1] == 'static': TASKS = TASKS[:2]
step('1 load raw files', ['load_raw.py'], ['andmal_static_X.npy', 'andmal_dyn.pkl', 'apt_flow.pkl'])
step('2 parse Unraveled logs', ['prep_unraveled.py'], ['unraveled_events.pkl'])
step('3 adapters + SABP', ['prep_all.py'], ['task_unraveled.npz'])
for t, k in TASKS:
    step(f'4 baselines {t} {k}', ['exp.py', t, k], [f'res_{t}_{k}.csv'])
    step(f'5 base learners {t} {k}', ['exp4.py', t, k], [f'L0_{t}_{k}_f4.npz'])
step('6 gate + meta-learner', ['meta.py'], None)
step('7 leakage + cross-domain', ['exp2.py'], ['res_universal.csv'])
step('8 summary, statistics, joint verdict', ['final.py'], None)
step('9 figures', ['figs.py'], None); step('9 framework figure', ['fig_framework.py'], None)
print('\nAll done. Send back from', config.WORK_DIR, ': summary.csv all_folds.csv stats.json joint.json gate.csv res_leakage.csv res_universal.csv res_unified_layer.csv and the info_*.pkl files')
