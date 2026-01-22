from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan

from src.env.urban_env import CarlaUrbanEnv


def _normalize_model_path(p: str) -> str:
    model_path = Path(p)
    if model_path.suffix.lower() != ".zip":
        model_path = model_path.with_suffix(".zip")
    return str(model_path).replace(".zip.zip", ".zip")


def _unwrap_to_base_env(env: Any):
    """
    Unwrap SB3 VecEnv/VecNormalize to the underlying gym env instance.
    """
    e = env
    # unwrap VecNormalize -> .venv
    while hasattr(e, "venv"):
        e = e.venv
    # unwrap DummyVecEnv -> .envs[0]
    if hasattr(e, "envs") and isinstance(e.envs, (list, tuple)) and len(e.envs) > 0:
        e = e.envs[0]
    # gymnasium unwrapped
    e = getattr(e, "unwrapped", e)
    return e


def _get_vehicle(env: Any):
    """
    Try common attribute names used in CARLA envs to find the ego vehicle actor.
    """
    base = _unwrap_to_base_env(env)
    for name in [
        "vehicle",
        "ego_vehicle",
        "_vehicle",
        "_ego_vehicle",
        "ego",
        "_ego",
        "actor",
        "_actor",
    ]:
        v = getattr(base, name, None)
        if v is not None:
            return v
    return None


def _speed_mps(env: Any, info: dict) -> Optional[float]:
    """
    Prefer speed from info; fallback to CARLA vehicle velocity magnitude.
    """
    if isinstance(info, dict) and "speed_mps" in info:
        try:
            return float(info["speed_mps"])
        except Exception:
            pass

    veh = _get_vehicle(env)
    if veh is None:
        return None

    try:
        v = veh.get_velocity()
        return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))
    except Exception:
        return None


def _control(env: Any):
    """
    Prefer last control cached by env (self._last_control).
    Fallback to CARLA vehicle.get_control().
    """
    base = _unwrap_to_base_env(env)

    ctrl = getattr(base, "_last_control", None)
    if ctrl is not None:
        return ctrl

    veh = _get_vehicle(env)
    if veh is None:
        return None

    try:
        return veh.get_control()
    except Exception:
        return None


def _pick_metrics(info: dict) -> dict:
    keys = [
        "collision",
        "collisions",
        "lane_invasion",
        "red_light_violation",
        "route_completion",
        "distance",
        "avg_speed",
        "speed_mps",
        "stuck",
        "termination_reason",
        "reason",
        "done_reason",
        # shield debug
        "action_name",
        "proposed_action_name",
        "action_overridden",
        "override_reason",
        # monitor (if present)
        "TimeLimit.truncated",
        "episode",
    ]
    if not isinstance(info, dict):
        return {}
    return {k: info[k] for k in keys if k in info}


def _extract_step_num(p: Path) -> int:
    # vecnormalize_50000.pkl -> 50000
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def _auto_find_vecnormalize(model_zip: Path) -> Optional[Path]:
    """
    Try to find VecNormalize stats near the model.
    Priority:
      1) vecnormalize_final.pkl in parent dirs
      2) latest vecnormalize_*.pkl by step number
    """
    candidates_dirs = [
        model_zip.parent,            # e.g. checkpoints/best_model
        model_zip.parent.parent,     # e.g. checkpoints
        Path("checkpoints"),         # fallback
    ]

    # 1) final
    for d in candidates_dirs:
        if not d.exists():
            continue
        p = d / "vecnormalize_final.pkl"
        if p.exists():
            return p

    # 2) latest by step
    best: Optional[Tuple[int, Path]] = None
    for d in candidates_dirs:
        if not d.exists():
            continue
        for p in d.glob("vecnormalize_*.pkl"):
            step = _extract_step_num(p)
            if best is None or step > best[0]:
                best = (step, p)

    return best[1] if best else None


def make_env(
    carla_cfg_path: str,
    sensors_cfg_path: str,
    scenarios_cfg_path: str,
    rl_cfg_path: str,
    seed: int,
    disable_shield: bool,
):
    env = CarlaUrbanEnv(
        carla_config_path=carla_cfg_path,
        sensors_config_path=sensors_cfg_path,
        scenarios_config_path=scenarios_cfg_path,
        rl_config_path=rl_cfg_path,
        seed=seed,
        disable_shield=disable_shield,
    )
    env = Monitor(env)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model (with or without .zip)")
    ap.add_argument("--carla", default="config/carla.yaml")
    ap.add_argument("--sensors", default="config/sensors.yaml")
    ap.add_argument("--scenarios", default="config/scenarios.yaml")
    ap.add_argument("--rl", default="config/rl.yaml")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep each step (slow motion)")
    ap.add_argument("--log-every", type=int, default=30, help="Print debug every N steps")
    ap.add_argument("--max-steps", type=int, default=0, help="Override max steps (0 = no override)")

    # disable shield
    ap.add_argument("--disable-shield", action="store_true", help="Disable env safety shield override")

    # ✅ NEW: load VecNormalize stats (optional)
    ap.add_argument("--vecnorm", default="", help="Path to VecNormalize .pkl (optional). If empty, auto-find.")

    args = ap.parse_args()

    model_path_str = _normalize_model_path(args.model)
    model_zip = Path(model_path_str)
    if not model_zip.exists():
        raise FileNotFoundError(f"Model not found: {model_zip}")

    # -------- build vec env like training --------
    venv = DummyVecEnv(
        [
            lambda: make_env(
                args.carla,
                args.sensors,
                args.scenarios,
                args.rl,
                seed=args.seed,
                disable_shield=bool(args.disable_shield),
            )
        ]
    )

    # override max_steps if needed (apply to base env)
    if args.max_steps > 0:
        base = _unwrap_to_base_env(venv)
        for attr in ["max_steps", "_max_steps", "episode_max_steps"]:
            if hasattr(base, attr):
                try:
                    setattr(base, attr, int(args.max_steps))
                except Exception:
                    pass

    # -------- VecNormalize load (if available) --------
    vecnorm_path: Optional[Path] = None
    if args.vecnorm.strip():
        p = Path(args.vecnorm.strip())
        if not p.exists():
            raise FileNotFoundError(f"VecNormalize stats not found: {p}")
        vecnorm_path = p
    else:
        vecnorm_path = _auto_find_vecnormalize(model_zip)

    if vecnorm_path is not None and vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        # freeze stats during eval
        venv.training = False
        venv.norm_reward = False
        print("Loaded VecNormalize stats:", vecnorm_path)
    else:
        print("VecNormalize stats: NOT used (no .pkl found).")

    # optional NaN checker in eval too
    venv = VecCheckNan(venv, raise_exception=True)

    # Load model with vec env
    model = PPO.load(str(model_zip), env=venv)
    print("Loaded model:", model_zip)
    print("disable_shield:", bool(args.disable_shield))

    # -------- eval loop --------
    for ep in range(args.episodes):
        obs = venv.reset()  # VecEnv reset -> obs dict with batch dim
        done = False
        ep_rew = 0.0
        steps = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = venv.step(action)

            # VecEnv shapes: rewards: (n_env,), dones: (n_env,), infos: list[dict]
            r0 = float(rewards[0])
            done0 = bool(dones[0])
            info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}

            ep_rew += r0
            steps += 1
            last_info = info0
            done = done0

            if args.log_every > 0 and (steps % args.log_every == 0 or done):
                spd = _speed_mps(venv, info0)
                ctrl = _control(venv)

                ctrl_str = ""
                if ctrl is not None:
                    t = getattr(ctrl, "throttle", None)
                    b = getattr(ctrl, "brake", None)
                    s = getattr(ctrl, "steer", None)
                    hb = getattr(ctrl, "hand_brake", None)
                    ctrl_str = f" | throttle={t} brake={b} steer={s} handbrake={hb}"

                sh = ""
                if isinstance(info0, dict):
                    sh = (
                        f" | proposed={info0.get('proposed_action_name')} "
                        f"executed={info0.get('action_name')} "
                        f"overridden={info0.get('action_overridden')} "
                        f"reason={info0.get('override_reason')}"
                    )

                # scalar action for print
                try:
                    a0 = int(action[0])
                except Exception:
                    try:
                        a0 = int(action)
                    except Exception:
                        a0 = action

                print(
                    f"[ep {ep+1} step {steps}] "
                    f"rew={r0:.3f} speed_mps={spd} action={a0} done={done0}"
                    f"{ctrl_str}{sh}"
                )

            if args.sleep > 0:
                time.sleep(args.sleep)

        metrics = _pick_metrics(last_info)
        print(f"Episode {ep+1}: steps={steps} reward={ep_rew:.2f} metrics={metrics}")
        print("Final info:", last_info)
        time.sleep(0.5)

    venv.close()


if __name__ == "__main__":
    main()
