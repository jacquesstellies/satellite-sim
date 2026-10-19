"""Parallel Monte Carlo driver for the nominal (nadir/sun pointing) mission profile.

Each run starts at rest (w = 0, as if just detumbled) from a uniformly random inertial
attitude and a random point in the orbit, with a static +-inertia_frac error on the
plant inertia (the controller keeps the nominal M_Inertia). Runs are independent and
seeded, so any single run can be reproduced with --only <run index>.

Outputs (in --out):
    run_XXX/            per-run decimated time series (csv), config and sim log
    summary.csv         one row of metrics per run
    aggregate.txt       pooled statistics across runs

Usage (from src/):
    python monte_carlo.py -c configs/EOS.toml -n 64 -w 6
"""
import os
# one BLAS thread per worker, otherwise numpy oversubscribes the cores
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import datetime
import multiprocessing as mp
import time
import traceback

import numpy as np
import pandas as pd
import toml
from scipy.spatial.transform import Rotation

import my_utils
from satellite import DivergentRate, DivergentAttitude, face_geometry
from simulation import Simulation

ROOT_DIR = os.popen("git rev-parse --show-toplevel").read().strip()
MU_EARTH = 3.986004418e14
R_EARTH = 6378.137e3

# columns kept in each run's csv
TS_COLUMNS = ['time', 'phase', 'err_ref_deg', 'err_target_deg', 'boresight_nadir_error_deg',
              'sun_axis_error_deg', 'w_sat_x', 'w_sat_y', 'w_sat_z',
              'w_sat_error_x', 'w_sat_error_y', 'w_sat_error_z', 'H_norm', 'eclipse', 'cos_factor']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default="configs/EOS.toml")
    p.add_argument("-n", "--runs", type=int, default=64)
    p.add_argument("-w", "--workers", type=int, default=6)
    p.add_argument("--orbits", type=float, default=1.5, help="run duration in orbital periods")
    p.add_argument("--duration", type=float, default=None, help="run duration in s (overrides --orbits)")
    p.add_argument("--seed", type=int, default=0, help="base seed; run i uses seed + i")
    p.add_argument("--inertia-frac", type=float, default=0.05, help="static plant inertia error, uniform +-frac per axis")
    p.add_argument("--log-dt", type=float, default=1.0, help="logging / output sample period (s)")
    p.add_argument("--out", default=None, help="output folder (default data_logs/mc_<config>_<date>)")
    p.add_argument("--only", type=int, nargs="*", default=None, help="run only these run indices")
    p.add_argument("--reanalyze", action="store_true",
                   help="recompute metrics from the run_*/ time series already in --out (no simulation)")
    # metric / failure thresholds
    p.add_argument("--acq-thresh", type=float, default=0.3, help="deg, error below which initial acquisition is complete")
    p.add_argument("--acq-hold", type=float, default=60.0, help="s the error must stay below --acq-thresh")
    p.add_argument("--ss-margin", type=float, default=300.0,
                   help="s after acquisition before steady state starts (lets the acquisition tail decay)")
    p.add_argument("--trans-window", type=float, default=600.0, help="s after a mode switch excluded from steady state")
    p.add_argument("--settle-thresh", type=float, default=0.2, help="deg, transition settling band")
    p.add_argument("--fail-thresh", type=float, default=5.0, help="deg, steady-state error that counts as loss of stability")
    p.add_argument("--w-max-dps", type=float, default=10.0, help="body rate that aborts a run as divergent")
    p.add_argument("--q-err-max-deg", type=float, default=10.0, help="pointing error to the commanded ref that aborts a run")
    p.add_argument("--q-err-persist", type=float, default=10.0, help="s the pointing error must persist before aborting")
    return p.parse_args()


def orbit_period(config):
    a = R_EARTH + config['orbit']['altitude_km'] * 1e3
    return 2 * np.pi * np.sqrt(a**3 / MU_EARTH)


def random_quat_xyzw(rng):
    """Uniformly distributed attitude (Shoemake), scalar-last."""
    x1, x2, x3 = rng.random(3)
    q = np.array([np.sqrt(1 - x1) * np.sin(2 * np.pi * x2),
                  np.sqrt(1 - x1) * np.cos(2 * np.pi * x2),
                  np.sqrt(x1) * np.sin(2 * np.pi * x3),
                  np.sqrt(x1) * np.cos(2 * np.pi * x3)])
    return q / np.linalg.norm(q)


def perturbed_inertia(J, frac, rng):
    """J_true = S J S with S = diag(sqrt(s)), s_i ~ U(1-frac, 1+frac): scales each
    principal moment by exactly s_i and keeps J symmetric positive definite."""
    s = rng.uniform(1 - frac, 1 + frac, 3)
    S = np.diag(np.sqrt(s))
    return S @ J @ S, s


def make_run_config(base_config, args, run_idx):
    rng = np.random.default_rng(args.seed + run_idx)
    config = copy.deepcopy(base_config)

    q_init = random_quat_xyzw(rng)
    mean_anomaly = rng.uniform(0, 360)
    J = np.array(config['satellite']['M_Inertia'], dtype=float)
    if J.shape == (3,):
        J = np.diag(J)
    J_plant, s = perturbed_inertia(J, args.inertia_frac, rng)

    config['satellite']['euler_init_en'] = False
    config['satellite']['q_init'] = q_init.tolist()
    config['satellite']['w_init_dps'] = [0.0, 0.0, 0.0]
    config['satellite']['M_Inertia_plant'] = J_plant.tolist()
    config['orbit']['mean_anomaly_deg'] = float(mean_anomaly)
    config['simulation']['duration'] = args.duration if args.duration else args.orbits * orbit_period(config)
    config['simulation']['iterations'] = 1
    config['simulation']['verbose'] = False
    config['simulation']['test_mode_en'] = False
    config['output']['log_enable'] = True
    config['output']['log_t_sample'] = args.log_dt
    config['output']['visualizer']['enable'] = False
    config['output']['show_plots'] = False
    config['FDIR']['satellite']['enable'] = True
    config['FDIR']['satellite']['w_max_dps'] = args.w_max_dps
    config['FDIR']['satellite']['q_err_max_deg'] = args.q_err_max_deg
    config['FDIR']['satellite']['q_err_persist_s'] = args.q_err_persist

    params = {'run': run_idx, 'seed': args.seed + run_idx,
              'q0_x': q_init[0], 'q0_y': q_init[1], 'q0_z': q_init[2], 'q0_w': q_init[3],
              'mean_anomaly_deg': mean_anomaly,
              'J_scale_x': s[0], 'J_scale_y': s[1], 'J_scale_z': s[2]}
    return config, params


def quat_angle_deg(q1, q2):
    """Angle between attitudes given as (N,4) arrays in the same component order."""
    dot = np.abs(np.sum(q1 * q2, axis=1))
    return np.degrees(2 * np.arccos(np.clip(dot, -1.0, 1.0)))


def postprocess(sim, config):
    """Adds per-sample metric columns to sim.results_df and returns it."""
    df = sim.results_df
    xyzw = ['x', 'y', 'z', 'w']
    q_sat = df[[f'q_sat_{a}' for a in xyzw]].to_numpy()
    night = df['q_nadir_ref_w'].notna().to_numpy()
    day = df['q_sun_ref_w'].notna().to_numpy()
    df['phase'] = np.where(night, 'night', np.where(day, 'day', 'none'))

    # |w|: q and -q are the same attitude (euler_axis_sat_error_deg reads ~360 deg when
    # the error quaternion lands in the w < 0 hemisphere)
    df['err_ref_deg'] = np.degrees(2 * np.arccos(np.clip(np.abs(df['q_sat_error_w']), 0.0, 1.0)))
    q_target = np.where(night[:, None], df[[f'q_nadir_ref_{a}' for a in xyzw]].to_numpy(),
                        df[[f'q_sun_ref_{a}' for a in xyzw]].to_numpy())
    df['err_target_deg'] = quat_angle_deg(q_sat, q_target)

    # sun-pointing axis vs sun direction (same construction as boresight_nadir_error_deg)
    r_sat = Rotation.from_quat(q_sat).inv()  # passive T_BI
    sun_axis = np.array(config['satellite'].get('nominal_day_sun_axis', [1.0, 0.0, 0.0]), dtype=float)
    sun_axis /= np.linalg.norm(sun_axis)
    n_sun_B = r_sat.apply(df[[f'n_sun_{a}' for a in my_utils.xyz_axes]].to_numpy())
    df['sun_axis_error_deg'] = np.degrees(np.arccos(np.clip(n_sun_B @ sun_axis, -1.0, 1.0)))
    return df


def ideal_cos_factor(config):
    """Area-weighted panel cos factor with the sun exactly on the sun-pointing axis,
    i.e. the best nominal_day can do given the panel cant."""
    panels = [f for f in face_geometry(config['satellite']['dimensions']['model'], config['satellite']['dimensions'])
              if f.solar_panel]
    if not panels:
        return np.nan
    sun_axis = np.array(config['satellite'].get('nominal_day_sun_axis', [1.0, 0.0, 0.0]), dtype=float)
    sun_axis /= np.linalg.norm(sun_axis)
    areas = np.array([f.area for f in panels])
    cos = np.array([max(0.0, f.norm_vec @ sun_axis / np.linalg.norm(f.norm_vec)) for f in panels])
    return areas @ cos / np.sum(areas)


def stats(x, prefix):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {f'{prefix}_rms': np.nan, f'{prefix}_p95': np.nan, f'{prefix}_max': np.nan}
    return {f'{prefix}_rms': np.sqrt(np.mean(x**2)), f'{prefix}_p95': np.percentile(x, 95), f'{prefix}_max': np.max(x)}


def compute_metrics(df, config, args):
    t = df['time'].to_numpy()
    err = df['err_ref_deg'].to_numpy()
    phase = df['phase'].to_numpy()
    m = {}

    # initial acquisition: first time after which err stays below acq_thresh for acq_hold.
    # The error to the raw target must be small too, else a ref still being smoothed
    # towards the target after the initial jump would count as acquired.
    below = (err < args.acq_thresh) & (df['err_target_deg'].to_numpy() < args.acq_thresh)
    hold_n = max(1, int(round(args.acq_hold / args.log_dt)))
    t_acq = np.nan
    run = 0
    for k, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= hold_n:
            t_acq = t[k - hold_n + 1]
            break
    m['t_acq'] = t_acq
    m['acquired'] = bool(np.isfinite(t_acq))

    # mode switches after acquisition
    sw_idx = np.flatnonzero(phase[1:] != phase[:-1]) + 1
    sw_idx = [k for k in sw_idx if np.isfinite(t_acq) and t[k] > t_acq]
    excluded = np.zeros(len(t), dtype=bool)
    transitions = []
    for k in sw_idx:
        win = (t >= t[k]) & (t < t[k] + args.trans_window)
        excluded |= win
        e_win = err[win]
        t_win = t[win]
        # window cut off by the end of the run: peak still counts, settling is unknown
        truncated = t[k] + args.trans_window > t[-1]
        outside = np.flatnonzero(e_win >= args.settle_thresh)
        settle = (t_win[outside[-1]] - t[k] + args.log_dt) if len(outside) else 0.0
        if len(outside) and outside[-1] == len(e_win) - 1:
            settle = np.nan  # not settled within the window
        transitions.append({'to': phase[k], 't': t[k], 'peak': np.max(e_win), 'settle': settle,
                            'truncated': truncated})
    m['n_transitions'] = len(transitions)
    for kind in ('day', 'night'):
        tr = [x for x in transitions if x['to'] == kind and not x['truncated']]
        m[f'trans_to_{kind}_peak_deg'] = max((x['peak'] for x in tr), default=np.nan)
        settles = [x['settle'] for x in tr]
        m[f'trans_to_{kind}_settle_s'] = (np.nan if not settles or any(np.isnan(settles)) else max(settles))
        m[f'trans_to_{kind}_unsettled'] = int(sum(np.isnan(s) for s in settles))

    ss = (t > t_acq + args.ss_margin) & ~excluded if np.isfinite(t_acq) else np.zeros(len(t), dtype=bool)
    m['ss_duration_s'] = np.sum(ss) * args.log_dt
    m.update(stats(err[ss], 'ss_err_ref'))
    m.update(stats(df['err_target_deg'].to_numpy()[ss], 'ss_err_target'))
    for kind in ('night', 'day'):
        mask = ss & (phase == kind)
        m[f'ss_{kind}_duration_s'] = np.sum(mask) * args.log_dt
        m.update(stats(err[mask], f'ss_{kind}_err_ref'))
    m.update(stats(df['boresight_nadir_error_deg'].to_numpy()[ss & (phase == 'night')], 'ss_night_boresight'))
    m.update(stats(df['sun_axis_error_deg'].to_numpy()[ss & (phase == 'day')], 'ss_day_sun_axis'))

    w_max = config['wheels']['max_speed_rpm'] * 2 * np.pi / 60
    w_wheels = np.abs(df[[c for c in df.columns if c.startswith('w_wheels_') and c[9:].isdigit()]].to_numpy())
    m['wheel_speed_max_frac'] = np.max(w_wheels) / w_max
    m['wheel_sat_frac_ss'] = np.mean(np.any(w_wheels[ss] > 0.98 * w_max, axis=1)) if np.any(ss) else np.nan

    # solar panel cos factor integrated over the last full orbit of the run
    if 'cos_factor' in df:
        T_orb = orbit_period(config)
        win = t >= t[-1] - T_orb
        c = df['cos_factor'].to_numpy()[win]
        sunlit = 1.0 - df['eclipse'].to_numpy()[win]
        m['cos_int_orbit_s'] = np.trapz(c, t[win])
        m['cos_avg_orbit'] = m['cos_int_orbit_s'] / (t[win][-1] - t[win][0])
        m['sunlit_frac_orbit'] = np.mean(sunlit)
        c_ideal = ideal_cos_factor(config)
        m['cos_int_ideal_s'] = c_ideal * np.trapz(sunlit, t[win])
        m['cos_eff'] = m['cos_int_orbit_s'] / m['cos_int_ideal_s']
        m['cos_window_after_acq'] = bool(np.isfinite(t_acq) and t[win][0] > t_acq)

    m['ss_exceed'] = bool(np.any(err[ss] > args.fail_thresh))
    m['n_ss_exceed_events'] = int(np.sum(np.diff((err[ss] > args.fail_thresh).astype(int)) == 1)
                                  + (err[ss][0] > args.fail_thresh if np.any(ss) else 0))
    return m, transitions


def run_one(task):
    base_config, args, run_idx, out_dir = task
    config, params = make_run_config(base_config, args, run_idx)
    run_name = f"run_{run_idx:03d}"
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"{run_name}_config.toml"), "w") as f:
        toml.dump(config, f)

    row = dict(params, status='ok', error='', wall_s=np.nan)
    t0 = time.time()
    try:
        sim = Simulation(config, {}, log_file_name=run_name, log_folder_path=run_dir, logging_en=True)
        sol = sim.simulate()
        sim.collect_results(sol)
        df = postprocess(sim, config)
        if not np.all(np.isfinite(df['err_ref_deg'])):
            row['status'] = 'nan'
        metrics, transitions = compute_metrics(df, config, args)
        row.update(metrics)
        if row['status'] == 'ok' and (not metrics['acquired'] or metrics['ss_exceed']):
            row['status'] = 'no_acq' if not metrics['acquired'] else 'ss_exceed'
        cols = [c for c in TS_COLUMNS if c in df] + [c for c in df.columns if c.startswith('cos_panel_')] \
            + [c for c in df.columns if c.startswith('w_wheels_') and c[9:].isdigit()]
        df[cols].to_csv(os.path.join(run_dir, f"{run_name}_ts.csv"), index=False)
        for tr in transitions:
            tr['run'] = run_idx
    except DivergentRate as e:
        row['status'], row['error'], transitions = 'diverged', str(e), []
    except DivergentAttitude as e:
        row['status'], row['error'], transitions = 'pointing_fault', str(e), []
    except Exception as e:
        row['status'], row['error'], transitions = 'error', repr(e), []
        with open(os.path.join(run_dir, "traceback.txt"), "w") as f:
            f.write(traceback.format_exc())
    row['wall_s'] = time.time() - t0
    return row, transitions


def reanalyze_one(task):
    base_config, args, run_idx, out_dir = task
    config, params = make_run_config(base_config, args, run_idx)
    run_name = f"run_{run_idx:03d}"
    ts_path = os.path.join(out_dir, run_name, f"{run_name}_ts.csv")
    row = dict(params, status='ok', error='', wall_s=np.nan)
    if not os.path.exists(ts_path):
        row['status'] = 'missing'  # diverged/errored run, or never ran
        return row, []
    df = pd.read_csv(ts_path)
    metrics, transitions = compute_metrics(df, config, args)
    row.update(metrics)
    if not np.all(np.isfinite(df['err_ref_deg'])):
        row['status'] = 'nan'
    elif not metrics['acquired'] or metrics['ss_exceed']:
        row['status'] = 'no_acq' if not metrics['acquired'] else 'ss_exceed'
    for tr in transitions:
        tr['run'] = run_idx
    return row, transitions


def wilson_upper(k, n, z=1.96):
    """Upper bound of the 95% Wilson score interval for k failures out of n."""
    if n == 0:
        return np.nan
    p = k / n
    centre = p + z**2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (centre + half) / (1 + z**2 / n)


def aggregate(summary, transitions, args, out_dir):
    n = len(summary)
    fails = summary[summary['status'] != 'ok']
    lines = [f"Monte Carlo summary ({datetime.datetime.now():%Y-%m-%d %H:%M})",
             f"config: {args.config}   runs: {n}   inertia error: +-{100*args.inertia_frac:.1f}%",
             f"status counts: {summary['status'].value_counts().to_dict()}",
             f"failures: {len(fails)}/{n}  (95% upper bound on failure rate: {100*wilson_upper(len(fails), n):.1f}%)",
             ""]
    ok = summary[summary['acquired'] == True] if 'acquired' in summary else summary.iloc[0:0]

    def describe(col, unit):
        if col not in ok or ok[col].dropna().empty:
            return f"  {col:32s} n/a"
        x = ok[col].dropna()
        return (f"  {col:32s} median {x.median():9.4f}  mean {x.mean():9.4f}  "
                f"p95 {x.quantile(0.95):9.4f}  worst {x.max():9.4f} {unit}")

    lines.append("Per-run metrics across acquired runs:")
    for col, unit in [('t_acq', 's'), ('ss_err_ref_rms', 'deg'), ('ss_err_ref_p95', 'deg'), ('ss_err_ref_max', 'deg'),
                      ('ss_night_err_ref_rms', 'deg'), ('ss_day_err_ref_rms', 'deg'),
                      ('ss_night_boresight_rms', 'deg'), ('ss_night_boresight_max', 'deg'),
                      ('ss_day_sun_axis_rms', 'deg'), ('ss_day_sun_axis_max', 'deg'),
                      ('ss_err_target_rms', 'deg'), ('wheel_speed_max_frac', '-'),
                      ('cos_int_orbit_s', 's'), ('cos_avg_orbit', '-'), ('cos_eff', '-'), ('sunlit_frac_orbit', '-'),
                      ('wall_s', 's')]:
        lines.append(describe(col, unit))

    if transitions:
        tr = pd.DataFrame(transitions)
        lines.append("")
        lines.append(f"Mode transitions (window {args.trans_window:.0f} s, settle band {args.settle_thresh} deg):")
        for kind, g in tr.groupby('to'):
            full = g[~g['truncated']]
            unsettled = int(full['settle'].isna().sum())
            lines.append(f"  -> {kind:5s}  n={len(g):3d}  peak median {g['peak'].median():.3f} worst {g['peak'].max():.3f} deg   "
                         f"settle median {full['settle'].median():.0f} worst {full['settle'].max():.0f} s   "
                         f"unsettled {unsettled}  (truncated by run end: {len(g) - len(full)})")
        tr.to_csv(os.path.join(out_dir, "transitions.csv"), index=False)

    if len(fails):
        lines.append("")
        lines.append("Failed runs:")
        for _, r in fails.iterrows():
            lines.append(f"  run {int(r['run']):3d} seed {int(r['seed'])}: {r['status']} {r.get('error', '')}")

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "aggregate.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


def main():
    args = parse_args()
    with open(args.config) as f:
        base_config = toml.load(f)

    if args.out is None:
        name = os.path.splitext(os.path.basename(args.config))[0]
        args.out = os.path.join(ROOT_DIR, "data_logs", f"mc_{name}_{datetime.datetime.now():%Y-%m-%d_%H-%M}")
    os.makedirs(args.out, exist_ok=True)
    indices = args.only if args.only else list(range(args.runs))
    tasks = [(base_config, args, i, args.out) for i in indices]

    if args.reanalyze:
        # same seeds/config as the original batch, so params and metrics line up per run
        results = [reanalyze_one(task) for task in tasks]
        # record the thresholds used, so plot_monte_carlo.py applies the same ones
        with open(os.path.join(args.out, "mc_args.toml"), "w") as f:
            toml.dump(vars(args), f)
        summary = pd.DataFrame([r for r, _ in results]).sort_values('run')
        summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
        aggregate(summary, [x for _, tr in results for x in tr], args, args.out)
        return

    with open(os.path.join(args.out, "mc_args.toml"), "w") as f:
        toml.dump(vars(args), f)
    print(f"{len(tasks)} runs x {(args.duration or args.orbits * orbit_period(base_config)):.0f} s on {args.workers} workers -> {args.out}")

    rows, transitions = [], []
    t0 = time.time()
    # maxtasksperchild=1: a fresh process per run, so nothing (memory, module state) leaks between runs
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=1) as pool:
        for row, tr in pool.imap_unordered(run_one, tasks):
            rows.append(row)
            transitions.extend(tr)
            print(f"[{len(rows):3d}/{len(tasks)}] run {row['run']:3d}: {row['status']:9s} "
                  f"ss rms {row.get('ss_err_ref_rms', np.nan):.4f} deg, max {row.get('ss_err_ref_max', np.nan):.4f} deg "
                  f"({row['wall_s']/60:.1f} min, elapsed {(time.time()-t0)/60:.1f} min)", flush=True)
            # write as we go so a long batch can be inspected (or survives a crash)
            pd.DataFrame(rows).sort_values('run').to_csv(os.path.join(args.out, "summary.csv"), index=False)

    summary = pd.DataFrame(rows).sort_values('run')
    aggregate(summary, transitions, args, args.out)


if __name__ == "__main__":
    main()
