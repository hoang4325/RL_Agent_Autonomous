from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO

from src.env.urban_env import CarlaUrbanEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to .zip model')
    ap.add_argument('--carla', default='config/carla.yaml', help='Path to carla.yaml')
    ap.add_argument('--sensors', default='config/sensors.yaml', help='Path to sensors.yaml')
    ap.add_argument('--scenarios', default='config/scenarios.yaml', help='Path to scenarios.yaml')
    ap.add_argument('--rl', default='config/rl.yaml', help='Path to rl.yaml')
    ap.add_argument('--episodes', type=int, default=3)
    args = ap.parse_args()

    env = CarlaUrbanEnv(
        carla_config_path=args.carla,
        sensors_config_path=args.sensors,
        scenarios_config_path=args.scenarios,
        rl_config_path=args.rl,
    )
    model = PPO.load(args.model)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            ep_rew += rew
            done = terminated or truncated
        print(f"Episode {ep+1}: reward={ep_rew:.2f} info={info}")
        time.sleep(1.0)

    env.close()


if __name__ == '__main__':
    main()
