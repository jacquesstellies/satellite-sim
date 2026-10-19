"""Figures for a monte_carlo.py batch.

Usage (from src/):
    python plot_monte_carlo.py ../data_logs/mc_EOS_pid_interp300

Writes into <out>/graphs/:
    mc_acquisition.png      error to target from the random initial attitude, all runs
    mc_transitions.png      error after each mode switch, aligned on the switch
    mc_ss_error_cdf.png     pooled steady-state error distributions, night vs day
    mc_per_run.png          per-run steady-state RMS / max error
    mc_cos_factor.png       panel cos factor through the day, and per-run orbit average
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import toml
import matplotlib.pyplot as plt

import my_utils  # thesis rcParams (fonts/sizes)
from monte_carlo import ideal_cos_factor, orbit_period

FIG_SIZE = (12, 7)
C_DAY = '#eb6834'
C_NIGHT = '#2a78d6'
C_RUN = '#9a9a9a'   # individual runs: recessive neutral
C_INK = '#222222'   # medians / reference lines
PHASE_COLOR = {'day': C_DAY, 'night': C_NIGHT}


def load(out_dir):
    summary = pd.read_csv(os.path.join(out_dir, 'summary.csv'))
    tr_path = os.path.join(out_dir, 'transitions.csv')
    transitions = pd.read_csv(tr_path) if os.path.exists(tr_path) else pd.DataFrame(columns=['to', 't', 'peak', 'settle', 'truncated', 'run'])
    args = toml.load(os.path.join(out_dir, 'mc_args.toml'))
    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, 'run_*', 'run_*_ts.csv'))):
        idx = int(os.path.basename(path)[4:7])
        runs[idx] = pd.read_csv(path)
    config_path = sorted(glob.glob(os.path.join(out_dir, 'run_*', 'run_*_config.toml')))[0]
    return summary, transitions, args, runs, toml.load(config_path)


def aligned(runs, events, column, t_before, t_after, dt):
    """Stack `column` of each run on a common time-since-event grid."""
    grid = np.arange(-t_before, t_after + dt / 2, dt)
    rows = []
    for run, t0 in events:
        df = runs[run]
        if t0 + t_after > df['time'].iloc[-1] or t0 - t_before < 0:
            continue  # event too close to either end of the run
        rows.append(np.interp(t0 + grid, df['time'], df[column]))
    return grid, np.array(rows)


def finish(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, axis='both', alpha=0.4)


def plot_acquisition(runs, summary, transitions, args, graphs):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    t_max = max(1000.0, np.nanpercentile(summary['t_acq'], 100) * 1.5)
    grid = np.arange(0, t_max, args['log_dt'])
    stack = []
    for run, df in runs.items():
        # stop each run at its first mode switch after acquisition: the target then
        # jumps to the other mode's attitude, which is a transition, not acquisition
        t_end = transitions.loc[transitions['run'] == run, 't'].min()
        t_end = min(t_max, t_end) if np.isfinite(t_end) else t_max
        m = df['time'] < t_end
        ax.plot(df['time'][m], df['err_target_deg'][m], color=C_RUN, lw=0.8, alpha=0.5)
        row = np.interp(grid, df['time'], df['err_target_deg'])
        row[grid >= t_end] = np.nan
        stack.append(row)
    ax.plot(grid, np.nanmedian(stack, axis=0), color=C_INK, lw=2, label='Median')
    ax.plot([], [], color=C_RUN, lw=0.8, label=f'Individual runs ($n={len(runs)}$)')
    ax.axhline(args['acq_thresh'], color=C_INK, ls='--', lw=1, label=f"Acquisition threshold ({args['acq_thresh']}$^\\circ$)")
    ax.set_yscale('log')
    ax.set_xlim(0, t_max)
    finish(ax, 'Time (s)', r'Error to target attitude $\theta_e$ (deg)')
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(graphs, 'mc_acquisition.png'), bbox_inches='tight')
    plt.close(fig)


def plot_transitions(runs, transitions, args, graphs):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_SIZE[0], FIG_SIZE[1] * 0.75))
    for ax, kind, title in zip(axes, ('day', 'night'), ('Night $\\to$ day (sun pointing)', 'Day $\\to$ night (nadir pointing)')):
        events = [(int(r.run), r.t) for r in transitions[transitions['to'] == kind].itertuples()]
        grid, stack = aligned(runs, events, 'err_ref_deg', 60, args['trans_window'], args['log_dt'])
        for row in stack:
            ax.plot(grid, row, color=C_RUN, lw=0.8, alpha=0.4)
        if len(stack):
            ax.plot(grid, np.median(stack, axis=0), color=PHASE_COLOR[kind], lw=2.2, label=f'Median ($n={len(stack)}$)')
        ax.axhline(args['settle_thresh'], color=C_INK, ls='--', lw=1, label=f"Settling band ({args['settle_thresh']}$^\\circ$)")
        ax.axvline(0, color=C_INK, lw=0.8)
        ax.set_yscale('log')
        if len(stack):
            ax.set_ylim(1e-2, 3 * np.max(stack))  # headroom above the peak for the legend
        ax.set_title(title)
        finish(ax, 'Time since mode switch (s)', r'Error to reference $\theta_e$ (deg)')
        ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(graphs, 'mc_transitions.png'), bbox_inches='tight')
    plt.close(fig)


def steady_state_samples(runs, summary, transitions, args):
    """Pooled steady-state samples, using the same exclusion rule as monte_carlo.py."""
    t_acq = summary.set_index('run')['t_acq']
    parts = []
    for run, df in runs.items():
        if run not in t_acq or not np.isfinite(t_acq[run]):
            continue
        ss = df['time'] > t_acq[run] + args.get('ss_margin', 0.0)
        for t0 in transitions.loc[transitions['run'] == run, 't']:
            ss &= ~((df['time'] >= t0) & (df['time'] < t0 + args['trans_window']))
        parts.append(df[ss])
    return pd.concat(parts) if parts else pd.DataFrame()


def ecdf(ax, x, color, label):
    x = np.sort(np.asarray(x)[np.isfinite(x)])
    if len(x) == 0:
        return
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, lw=2, label=label)
    p95 = np.percentile(x, 95)
    ax.plot([p95], [0.95], 'o', ms=8, color=color, markeredgecolor='white', markeredgewidth=2)


def plot_ss_cdf(ss, graphs):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_SIZE[0], FIG_SIZE[1] * 0.75), sharey=True)
    for kind in ('night', 'day'):
        ecdf(axes[0], ss.loc[ss['phase'] == kind, 'err_ref_deg'], PHASE_COLOR[kind],
             'Night (nadir)' if kind == 'night' else 'Day (sun)')
    axes[0].set_title('Full attitude error')
    finish(axes[0], r'Error to reference $\theta_e$ (deg)', 'Cumulative fraction of time')
    ecdf(axes[1], ss.loc[ss['phase'] == 'night', 'boresight_nadir_error_deg'], C_NIGHT, r'Boresight to nadir $\theta_{bn}$ (night)')
    ecdf(axes[1], ss.loc[ss['phase'] == 'day', 'sun_axis_error_deg'], C_DAY, r'Sun axis to sun $\theta_{s}$ (day)')
    axes[1].set_title('Pointing-axis error')
    finish(axes[1], 'Pointing error (deg)', '')
    for ax in axes:
        ax.axhline(0.95, color=C_INK, ls=':', lw=1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc='center right')
    fig.tight_layout()
    fig.savefig(os.path.join(graphs, 'mc_ss_error_cdf.png'), bbox_inches='tight')
    plt.close(fig)


def plot_per_run(summary, args, graphs):
    fig, ax = plt.subplots(figsize=(FIG_SIZE[0], FIG_SIZE[1] * 0.7))
    s = summary.sort_values('run')
    ok = s['status'] == 'ok'
    ax.plot(s['run'][ok], s['ss_err_ref_max'][ok], 'o', ms=8, color=C_NIGHT, markeredgecolor='white', markeredgewidth=1.5, label='Max')
    ax.plot(s['run'][ok], s['ss_err_ref_rms'][ok], 'o', ms=8, color=C_DAY, markeredgecolor='white', markeredgewidth=1.5, label='RMS')
    if (~ok).any():
        for r in s[~ok].itertuples():
            ax.axvline(r.run, color='#e34948', lw=1.5)
        ax.plot([], [], color='#e34948', lw=1.5, label='Failed run')
    ax.set_ylim(0, 1.3 * np.nanmax(s['ss_err_ref_max']))  # headroom for the legend
    finish(ax, 'Run', r'Steady-state error $\theta_e$ (deg)')
    ax.legend(loc='upper right', ncol=3)
    fig.savefig(os.path.join(graphs, 'mc_per_run.png'), bbox_inches='tight')
    plt.close(fig)


def plot_cos_factor(runs, summary, config, args, graphs):
    if not runs or 'cos_factor' not in next(iter(runs.values())):
        print("no cos_factor logged in this batch, skipping cos factor figure")
        return
    c_ideal = ideal_cos_factor(config)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_SIZE[0], FIG_SIZE[1] * 0.75), gridspec_kw={'width_ratios': [2.2, 1]})

    # (a) through each sunlit arc, aligned on eclipse exit (after acquisition)
    t_acq = summary.set_index('run')['t_acq']
    events = []
    for run, df in runs.items():
        exits = df['time'][(df['eclipse'].diff() == -1)]
        events += [(run, t) for t in exits if np.isfinite(t_acq.get(run, np.nan)) and t > t_acq[run]]
    ax = axes[0]
    t_after = 0.98 * summary['sunlit_frac_orbit'].min() * orbit_period(config)  # one sunlit arc
    grid, stack = aligned(runs, events, 'cos_factor', 120, t_after, args['log_dt'])
    for row in stack:
        ax.plot(grid, row, color=C_RUN, lw=0.8, alpha=0.4)
    if len(stack):
        ax.plot(grid, np.median(stack, axis=0), color=C_DAY, lw=2.2, label=f'Median ($n={len(stack)}$)')
    ax.axhline(c_ideal, color=C_INK, ls='--', lw=1, label=f'Ideal sun pointing ({c_ideal:.3f})')
    ax.axvline(0, color=C_INK, lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_title('Through the sunlit arc')
    finish(ax, 'Time since eclipse exit (s)', r'Panel cos factor $c$')
    ax.legend(loc='lower right')

    # (b) per-run orbit average vs the ideal for that orbit's sunlit fraction
    ax = axes[1]
    s = summary[summary['cos_avg_orbit'].notna()]
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(s))
    ax.plot(jitter, s['cos_eff'] * 100, 'o', ms=8, color=C_DAY, markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
    ax.axhline(100, color=C_INK, ls='--', lw=1, label='Ideal')
    lo = np.nanmin(s['cos_eff'] * 100)
    ax.set_ylim(lo - 0.3 * (100 - lo) - 0.05, 100 + 0.3 * (100 - lo) + 0.05)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_title('Per run, last full orbit')
    finish(ax, '', r'$\int c\,dt$ relative to ideal (%)')
    ax.legend(loc='center right')
    fig.tight_layout()
    fig.savefig(os.path.join(graphs, 'mc_cos_factor.png'), bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out_dir', help='monte_carlo.py output folder')
    a = p.parse_args()
    summary, transitions, args, runs, config = load(a.out_dir)
    graphs = os.path.join(a.out_dir, 'graphs')
    os.makedirs(graphs, exist_ok=True)

    plot_acquisition(runs, summary, transitions, args, graphs)
    plot_transitions(runs, transitions, args, graphs)
    plot_ss_cdf(steady_state_samples(runs, summary, transitions, args), graphs)
    plot_per_run(summary, args, graphs)
    plot_cos_factor(runs, summary, config, args, graphs)
    print(f"figures written to {graphs}")


if __name__ == '__main__':
    main()
