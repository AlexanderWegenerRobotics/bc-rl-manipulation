import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np

from src.env.pick_place_env import PickPlaceEnv
from src.policy.bc_policy import BCPolicy
from src.simulation.episode_logger import EpisodeLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a BC policy in the pick-place env.")
    parser.add_argument('--checkpoint', type=Path, default=Path('models/checkpoints/bc_policy.pt'))
    parser.add_argument('--config',     type=Path, default=Path('config/sim_config.yaml'))
    parser.add_argument('--episodes',   type=int,  default=20)
    parser.add_argument('--max-steps',  type=int,  default=1000)
    parser.add_argument('--render',     action='store_true')
    parser.add_argument('--device',     type=str,  default='cpu')
    parser.add_argument('--seed',       type=int,  default=0)
    parser.add_argument('--log-dir',    type=Path, default=None, help='If set, log rollouts to this directory')
    return parser.parse_args()


def run_episode(env: PickPlaceEnv, policy: BCPolicy, max_steps: int,
                logger: EpisodeLogger = None) -> dict:
    """Run one episode, log if logger provided, return result dict."""
    obs, info = env.reset()
    total_reward = 0.0
    success      = False

    raw = env.sim.get_obs()
    if logger:
        logger.start_episode(raw['pick_pos'], raw['place_pos'])

    for step in range(max_steps):
        action = policy.predict(obs)

        if logger:
            logger.log_step(raw, action)

        obs, reward, terminated, truncated, info = env.step(action)
        raw = env.sim.get_obs()
        total_reward += reward

        if terminated:
            success = True
            break
        if truncated:
            break

    if logger:
        logger.end_episode(success)

    return {
        'success':       success,
        'steps':         step + 1,
        'total_reward':  total_reward,
        'dist_to_place': info['dist_to_place'],
        'grasped':       info['grasped'],
    }


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config['simulation']['max_steps'] = args.max_steps
    render_mode = 'human' if args.render else None

    print(f"Loading policy from {args.checkpoint}")
    policy = BCPolicy.load(args.checkpoint, device=args.device)

    env = PickPlaceEnv(config, render_mode=render_mode)

    log_enabled = args.log_dir is not None
    logger = EpisodeLogger(str(args.log_dir), enabled=log_enabled) if log_enabled else EpisodeLogger("", enabled=False)

    results = []
    for ep in range(args.episodes):
        result = run_episode(env, policy, args.max_steps, logger=logger)
        results.append(result)
        status = "SUCCESS" if result['success'] else f"fail (dist={result['dist_to_place']:.3f}m)"
        print(f"  Episode {ep+1:3d}/{args.episodes} | {status} | "
              f"steps={result['steps']} | grasped={result['grasped']}")

    env.close()
    logger.close()

    successes    = [r['success'] for r in results]
    success_rate = np.mean(successes)
    mean_steps   = np.mean([r['steps'] for r in results])
    mean_dist    = np.mean([r['dist_to_place'] for r in results])
    grasped_rate = np.mean([r['grasped'] for r in results])

    print(f"\n{'='*50}")
    print(f"Episodes:       {args.episodes}")
    print(f"Success rate:   {success_rate*100:.1f}%  ({sum(successes)}/{args.episodes})")
    print(f"Grasp rate:     {grasped_rate*100:.1f}%")
    print(f"Mean steps:     {mean_steps:.0f}")
    print(f"Mean final dist:{mean_dist:.4f}m")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()