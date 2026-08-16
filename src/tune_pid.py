#!/usr/bin/env python3
"""Pole-place PID gains from wn, zeta, and sigma for a satellite config.

The controller (see controller.calc_pid_torque) commands wheel torque
    u = J (-kj q_v + kd w_e - ki int q_v)
so the satellite sees T_sat = -u. Small-angle quaternion kinematics
qdot_v ~= -0.5 w then give the characteristic polynomial

    (s + sigma)(s^2 + 2 zeta wn s + wn^2)

on each body axis after inertia cancellation. sigma=0 recovers PD.

Examples
--------
    python tune_pid.py --wn 0.15 --zeta 0.7
    python tune_pid.py -c config.toml --wn 0.2 --zeta 0.8 --sigma 0.02 --apply
    python tune_pid.py -c configs/EOS.toml --wn 0.1 --zeta 0.707 --sigma-ratio 0.1
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import toml
from scipy.spatial.transform import Rotation

import my_utils


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.toml"
DEFAULT_GAINS_OUT = SCRIPT_DIR / "configs" / "gains_pid.toml"


def load_inertia(config: dict) -> np.ndarray:
    J = np.array(config["satellite"]["M_Inertia"], dtype=float)
    if J.shape == (3, 3):
        return J
    if J.shape == (3,):
        return np.diag(J)
    raise ValueError("satellite.M_Inertia must be 3x3 or length-3")


def config_quaternions(config: dict) -> tuple[np.quaternion, np.quaternion]:
    sat = config["satellite"]
    if sat.get("euler_init_en", False):
        q_xyzw = Rotation.from_euler("xyz", sat["euler_init"], degrees=True).as_quat()
        q_BI = np.quaternion(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
    else:
        q_init = sat["q_init"]
        q_BI = np.quaternion(q_init[3], q_init[0], q_init[1], q_init[2])

    if sat.get("use_ref_euler", False):
        q_xyzw = Rotation.from_euler("xyz", sat["ref_euler"], degrees=True).as_quat()
        q_RI = np.quaternion(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
    elif sat.get("use_ref_q", False) or sat.get("use_ref_series", False):
        q_ref = sat["ref_q"] if sat.get("use_ref_q", False) else sat["ref_q_series"][0]
        q_RI = np.quaternion(q_ref[3], q_ref[0], q_ref[1], q_ref[2])
    else:
        q_RI = np.quaternion(1, 0, 0, 0)
    return q_BI.normalized(), q_RI.normalized()


def pid_gains(wn: float, zeta: float, sigma: float) -> tuple[float, float, float]:
    """Return (kj, kd, ki) for (s+sigma)(s^2 + 2 zeta wn s + wn^2)."""
    if wn <= 0:
        raise ValueError("wn must be > 0")
    if zeta <= 0:
        raise ValueError("zeta must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    kd = 2.0 * zeta * wn + sigma
    kj = 2.0 * (wn**2 + 2.0 * zeta * wn * sigma)
    ki = 2.0 * sigma * wn**2
    return kj, kd, ki


def second_order_metrics(wn: float, zeta: float) -> dict:
    ts = 4.0 / (zeta * wn) if zeta * wn > 0 else math.inf
    if zeta < 1.0:
        wd = wn * math.sqrt(1.0 - zeta**2)
        overshoot = math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta**2))
    else:
        wd = 0.0
        overshoot = 0.0
    return {
        "settling_s_2pct": ts,
        "damped_freq_rad_s": wd,
        "overshoot_frac": overshoot,
    }


def closed_loop_poles(wn: float, zeta: float, sigma: float) -> np.ndarray:
    # (s+sigma)(s^2 + 2 zeta wn s + wn^2)
    return np.roots([1.0, 2.0 * zeta * wn + sigma, wn**2 + 2.0 * zeta * wn * sigma, sigma * wn**2])


def peak_command(J: np.ndarray, kj: float, q_err: np.quaternion, D_pinv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_v = np.array([q_err.x, q_err.y, q_err.z])
    u_body = -kj * J @ q_v  # rest, w=0, integral=0
    u_wheels = D_pinv @ u_body
    return u_body, u_wheels


def allocation_matrix(config: dict) -> tuple[np.ndarray, np.ndarray]:
    layout = config["wheels"]["config"]
    if layout == "ortho":
        D = np.eye(3)
    elif layout == "pyramid":
        D = np.array([[-1, -1, 1, 1], [1, -1, -1, 1], [1, 1, 1, 1]], dtype=float)
    elif layout == "tetra":
        D = np.array(
            [[0.9428, -0.4714, -0.4714, 0],
             [0, 0.8165, -0.8165, 0],
             [-0.3333, -0.3333, -0.3333, 1]],
            dtype=float,
        )
    elif layout == "custom":
        D = np.array(config["wheels"]["D"], dtype=float)
    else:
        raise ValueError(f"unknown wheel layout {layout}")
    D_pinv = np.linalg.pinv(D) if D.shape[1] != 3 else np.linalg.inv(D)
    return D, D_pinv


def patch_controller_gains(config_path: Path, kj: float, kd: float, ki: float) -> None:
    text = config_path.read_text()
    section = re.search(r"(?ms)^\[controller\]\s*\n(.*?)(?=^\[|\Z)", text)
    if section is None:
        raise RuntimeError(f"no [controller] section in {config_path}")
    body = section.group(1)
    start, end = section.span(1)

    def replace_key(block: str, key: str, value: float) -> str:
        pattern = rf"^({re.escape(key)}\s*=\s*)[^\n#]+"
        repl = rf"\g<1>{value:.8g}"
        new_block, n = re.subn(pattern, repl, block, count=1, flags=re.M)
        if n != 1:
            raise RuntimeError(f"could not update {key} in [controller] of {config_path}")
        return new_block

    body = replace_key(body, "kj", kj)
    body = replace_key(body, "kd", kd)
    body = replace_key(body, "ki", ki)
    config_path.write_text(text[:start] + body + text[end:])


def write_gains_toml(
    path: Path,
    *,
    config_path: Path,
    wn: float,
    zeta: float,
    sigma: float,
    kj: float,
    kd: float,
    ki: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        f"# Generated by tune_pid.py — do not edit by hand; re-run the script.\n"
        f"# source_config = \"{config_path}\"\n"
        f"# closed-loop: (s + sigma)(s^2 + 2*zeta*wn*s + wn^2)\n"
        f"# kinematics factor of 2 is included (q_v ~= theta/2).\n\n"
        f"[pid_tuning]\n"
        f"wn = {wn:.8g}\n"
        f"zeta = {zeta:.8g}\n"
        f"sigma = {sigma:.8g}\n\n"
        f"[controller]\n"
        f"kj = {kj:.8g}\n"
        f"kd = {kd:.8g}\n"
        f"ki = {ki:.8g}\n"
    )
    path.write_text(contents)


def format_poles(poles: np.ndarray) -> str:
    parts = []
    for p in poles:
        if abs(p.imag) < 1e-12:
            parts.append(f"{p.real:.5g}")
        else:
            parts.append(f"{p.real:.5g}{p.imag:+.5g}j")
    return ", ".join(parts)


def resolve_spec(args, config: dict) -> tuple[float, float, float]:
    tuning = config.get("pid_tuning", {})
    wn = args.wn if args.wn is not None else tuning.get("wn")
    zeta = args.zeta if args.zeta is not None else tuning.get("zeta")
    if wn is None or zeta is None:
        raise SystemExit("wn and zeta are required (CLI --wn/--zeta or [pid_tuning] in the config)")

    if args.pd:
        sigma = 0.0
    elif args.sigma is not None:
        sigma = args.sigma
    elif args.sigma_ratio is not None:
        sigma = args.sigma_ratio * wn
    elif "sigma" in tuning:
        sigma = float(tuning["sigma"])
    else:
        sigma = 0.1 * wn
        print(f"sigma not given; using 0.1*wn = {sigma:.5g} rad/s (pass --pd for no integral)")
    return float(wn), float(zeta), float(sigma)


def print_report(
    *,
    config_path: Path,
    config: dict,
    wn: float,
    zeta: float,
    sigma: float,
    kj: float,
    kd: float,
    ki: float,
) -> None:
    J = load_inertia(config)
    evals = np.sort(np.linalg.eigvalsh(J))
    metrics = second_order_metrics(wn, zeta)
    poles = closed_loop_poles(wn, zeta, sigma)
    dt = float(config["controller"]["t_sample"])
    T_max = float(config["wheels"]["max_torque"])
    layout = config["wheels"]["config"]
    D, D_pinv = allocation_matrix(config)

    q_BI, q_RI = config_quaternions(config)
    q_err = my_utils.quat_error(q_RI, q_BI)
    prin_deg = my_utils.get_principal_angle_from_np_quaternion(q_err) * 180.0 / math.pi
    u_body, u_wheels = peak_command(J, kj, q_err, D_pinv)
    peak_wheel = float(np.max(np.abs(u_wheels))) if u_wheels.size else 0.0

    print()
    print(f"config      {config_path}")
    print(f"wheels      {layout}   T_max={T_max:g} Nm   dt={dt:g} s")
    print(f"J diag      {np.diag(J)}")
    print(f"J eigs      {evals}")
    print()
    print("spec")
    print(f"  wn        {wn:.6g} rad/s   ({wn * 180.0 / math.pi:.4g} deg/s)")
    print(f"  zeta      {zeta:.6g}")
    print(f"  sigma     {sigma:.6g} rad/s   ({'PD' if sigma == 0 else f'{sigma / wn:.3g} * wn'})")
    print()
    print("gains  (inertia-normalized; controller multiplies by J)")
    print(f"  kj        {kj:.8g}   1/s^2")
    print(f"  kd        {kd:.8g}   1/s")
    print(f"  ki        {ki:.8g}   1/s^3")
    print()
    print("physical Kp = kj * J_ii  (Nm / quat-vector ≈ 2x Nm/rad)")
    for axis, Jii in zip("xyz", np.diag(J)):
        print(f"  Kp_{axis}      {kj * Jii:.5g} Nm")
    print()
    print("linear response (PD pair; extra pole at -sigma)")
    print(f"  poles     {format_poles(poles)}")
    print(f"  ts(2%)    {metrics['settling_s_2pct']:.4g} s   (≈ 4/(zeta wn))")
    if zeta < 1.0:
        print(f"  Mp        {100.0 * metrics['overshoot_frac']:.2f} %")
        print(f"  wd        {metrics['damped_freq_rad_s']:.5g} rad/s")
    else:
        print("  Mp        0 % (overdamped)")
    print()
    print("feasibility")
    samples_per_period = (2.0 * math.pi / wn) / dt if wn > 0 else math.inf
    print(f"  samples / 2pi/wn period   {samples_per_period:.1f}   (want ≳ 20)")
    if samples_per_period < 20:
        print("  WARNING  wn is fast vs controller t_sample; drop wn or dt")
    print(f"  config attitude error     {prin_deg:.3g} deg")
    print(f"  |u_body| at rest          {np.abs(u_body)}")
    print(f"  |u_wheels| at rest        {np.abs(u_wheels)}")
    print(f"  peak |u_wheel| / T_max    {peak_wheel / T_max:.3g}" if T_max > 0 else "")
    if T_max > 0 and peak_wheel > T_max:
        print("  WARNING  initial error would saturate the wheels; lower wn or the slew")
    elif T_max > 0 and peak_wheel > 0.5 * T_max:
        print("  note     initial command uses >50% of wheel torque")
    if sigma > 0 and sigma > wn:
        print("  note     sigma > wn: integral pole is faster than the PD pair")
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tune_pid",
        description="Compute PID gains (kj, kd, ki) from wn, zeta, and sigma for a satellite config.",
    )
    p.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG, help="satellite toml (default: src/config.toml)")
    p.add_argument("--wn", type=float, default=None, help="natural frequency [rad/s]")
    p.add_argument("--zeta", type=float, default=None, help="damping ratio")
    p.add_argument("--sigma", type=float, default=None, help="real pole [rad/s]; 0 = PD")
    p.add_argument("--sigma-ratio", type=float, default=None, help="set sigma = ratio * wn (typical 0.1)")
    p.add_argument("--pd", action="store_true", help="force sigma=0 (no integral)")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_GAINS_OUT, help="gains toml to write")
    p.add_argument("--apply", action="store_true", help="also patch kj/kd/ki in the source config")
    p.add_argument("--dry-run", action="store_true", help="print only; do not write files")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config)
    if not config_path.is_file():
        alt = SCRIPT_DIR / args.config
        if alt.is_file():
            config_path = alt
        else:
            raise SystemExit(f"config not found: {args.config}")

    config = toml.load(config_path)
    wn, zeta, sigma = resolve_spec(args, config)
    kj, kd, ki = pid_gains(wn, zeta, sigma)
    print_report(
        config_path=config_path,
        config=config,
        wn=wn,
        zeta=zeta,
        sigma=sigma,
        kj=kj,
        kd=kd,
        ki=ki,
    )

    if args.dry_run:
        return 0

    out = args.output if args.output.is_absolute() else (Path.cwd() / args.output)
    write_gains_toml(out, config_path=config_path, wn=wn, zeta=zeta, sigma=sigma, kj=kj, kd=kd, ki=ki)
    print(f"wrote {out}")
    if args.apply:
        patch_controller_gains(config_path, kj, kd, ki)
        print(f"applied kj/kd/ki -> {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
