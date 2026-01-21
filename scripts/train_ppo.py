from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan

from src.utils.yaml_io import load_yaml
from src.env.urban_env import CarlaUrbanEnv
from src.rl.extractor import MultiModalExtractor


class CheckpointCallback(BaseCallback):
    def __init__(self, save_every_steps: int, save_dir: str):
        super().__init__()
        self.save_every_steps = int(save_every_steps)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.save_every_steps == 0:
            path = self.save_dir / f"ppo_urban_{self.num_timesteps}.zip"
            self.model.save(path)

            # If using VecNormalize, also checkpoint normalization statistics
            try:
                venv = self.model.get_env()
                if hasattr(venv, "save"):
                    venv.save(str(self.save_dir / f"vecnormalize_{self.num_timesteps}.pkl"))
            except Exception:
                pass
        return True


def make_env(
    carla_cfg_path: str,
    sensors_cfg_path: str,
    scenarios_cfg_path: str,
    rl_cfg_path: str,
    seed: int = 0,
):
    """Create a single CARLA env wrapped with Monitor."""
    env = CarlaUrbanEnv(
        carla_config_path=carla_cfg_path,
        sensors_config_path=sensors_cfg_path,
        scenarios_config_path=scenarios_cfg_path,
        rl_config_path=rl_cfg_path,
        seed=seed,
    )
    env = Monitor(env)

    # Seed once for reproducibility
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()

    return env


def _wrap_vecnormalize_if_enabled(env, rl_cfg, gamma: float):
    """Wrap VecNormalize if enabled. Prefer normalizing only 'vec' (not 'bev')."""
    norm_cfg = rl_cfg.get("normalize", {})
    use_norm = bool(norm_cfg.get("enabled", True))
    if not use_norm:
        return env, False

    clip_obs = float(norm_cfg.get("clip_obs", 10.0))
    clip_reward = float(norm_cfg.get("clip_reward", 10.0))
    norm_obs_keys = norm_cfg.get("norm_obs_keys", ["vec"])

    try:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
            norm_obs_keys=norm_obs_keys,  # SB3 newer versions
        )
    except TypeError:
        # Older SB3 versions without norm_obs_keys
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
        )

    # Explicit: training mode ON for the training env
    try:
        env.training = True
        env.norm_reward = True
    except Exception:
        pass

    return env, True


def _build_eval_env(args, rl_cfg, gamma: float, seed: int, train_env, use_norm: bool):
    """Create eval env; if VecNormalize used, share running stats from train env."""
    eval_env = DummyVecEnv([lambda: make_env(args.carla, args.sensors, args.scenarios, args.rl, seed=seed + 123)])

    if use_norm:
        # Create an eval VecNormalize wrapper (but don't update stats during eval)
        norm_cfg = rl_cfg.get("normalize", {})
        clip_obs = float(norm_cfg.get("clip_obs", 10.0))
        clip_reward = float(norm_cfg.get("clip_reward", 10.0))
        norm_obs_keys = norm_cfg.get("norm_obs_keys", ["vec"])

        try:
            eval_env = VecNormalize(
                eval_env,
                norm_obs=True,
                norm_reward=False,  # do NOT normalize rewards for eval reporting
                clip_obs=clip_obs,
                clip_reward=clip_reward,
                gamma=gamma,
                norm_obs_keys=norm_obs_keys,
            )
        except TypeError:
            eval_env = VecNormalize(
                eval_env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=clip_obs,
                clip_reward=clip_reward,
                gamma=gamma,
            )

        # Share stats from train env so eval obs normalization matches training
        try:
            eval_env.obs_rms = train_env.obs_rms
        except Exception:
            pass
        try:
            eval_env.ret_rms = train_env.ret_rms
        except Exception:
            pass

        # Freeze stats updates
        try:
            eval_env.training = False
            eval_env.norm_reward = False
        except Exception:
            pass

    # Catch NaNs in eval too (optional but helpful)
    eval_env = VecCheckNan(eval_env, raise_exception=True)
    return eval_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla", default="config/carla.yaml", help="Path to carla.yaml")
    ap.add_argument("--sensors", default="config/sensors.yaml", help="Path to sensors.yaml")
    ap.add_argument("--scenarios", default="config/scenarios.yaml", help="Path to scenarios.yaml")
    ap.add_argument("--rl", default="config/rl.yaml", help="Path to rl.yaml")
    args = ap.parse_args()

    rl_cfg = load_yaml(args.rl)
    total_steps = int(rl_cfg["rl"]["total_timesteps"])
    seed = int(rl_cfg["rl"].get("seed", 0))

    ppo_cfg = rl_cfg.get("ppo", {})
    log_cfg = rl_cfg.get("logging", {})
    tensorboard_log = log_cfg.get("tensorboard_log", "runs")
    ckpt_dir = log_cfg.get("checkpoint_dir", "checkpoints")

    gamma = float(ppo_cfg.get("gamma", 0.99))

    # --------- Train env ---------
    env = DummyVecEnv([lambda: make_env(args.carla, args.sensors, args.scenarios, args.rl, seed=seed)])

    # VecNormalize (recommended)
    env, use_norm = _wrap_vecnormalize_if_enabled(env, rl_cfg, gamma)

    # Catch NaNs/Inf early (very useful in CARLA + sensor pipelines)
    env = VecCheckNan(env, raise_exception=True)

    # --------- Eval env (optional but recommended) ---------
    eval_cfg = rl_cfg.get("eval", {})
    enable_eval = bool(eval_cfg.get("enabled", True))
    eval_freq = int(eval_cfg.get("eval_freq", 10000))
    n_eval_episodes = int(eval_cfg.get("n_eval_episodes", 5))
    deterministic_eval = bool(eval_cfg.get("deterministic", True))

    eval_env = None
    eval_cb = None
    if enable_eval:
        eval_env = _build_eval_env(args, rl_cfg, gamma, seed, env, use_norm)
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(Path(ckpt_dir) / "best_model"),
            log_path=str(Path(ckpt_dir) / "eval_logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic_eval,
        )

    # --------- PPO model ---------
    policy_kwargs = dict(
        features_extractor_class=MultiModalExtractor,
        normalize_images=False,  # IMPORTANT for BEV float tensors (avoid /255 style scaling)
    )

    model = PPO(
        rl_cfg["rl"].get("policy", "MultiInputPolicy"),
        env,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 256)),
        gamma=gamma,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        device="auto",
    )

    # --------- Callbacks ---------
    save_every = int(log_cfg.get("save_every_steps", 50000))
    cb = CheckpointCallback(save_every_steps=save_every, save_dir=ckpt_dir)

    callbacks = [cb]
    if eval_cb is not None:
        callbacks.append(eval_cb)
    callback = CallbackList(callbacks)

    model.learn(total_timesteps=total_steps, callback=callback)

    # Save final model + (if VecNormalize) stats
    final_path = Path(ckpt_dir) / "ppo_urban_final.zip"
    model.save(final_path)
    try:
        if hasattr(env, "save"):
            env.save(str(Path(ckpt_dir) / "vecnormalize_final.pkl"))
    except Exception:
        pass

    # Close envs
    try:
        if eval_env is not None:
            eval_env.close()
    except Exception:
        pass
    env.close()


if __name__ == "__main__":
    main()
