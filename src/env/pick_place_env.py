import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from src.simulation.sim import Simulation
from src.simulation.rendering import make_renderer
from src.robot.pose import Pose


PLACE_THRESHOLD = 0.05   # metres — object must be within this distance of place target
MAX_STEPS       = 1000   # episode timeout


class PickPlaceEnv(gym.Env):
    """
    Gymnasium environment for unimanual pick and place.

    Observation (22-dim):
        ee_pos      (3)  world frame
        ee_quat     (4)  wxyz world frame
        obj_pos     (3)  world frame
        obj_quat    (4)  wxyz
        pick_pos    (3)  world frame
        place_pos   (3)  world frame
        gripper_w   (1)
        mode        (1)  always 0 for unimanual

    Action (8-dim):
        ee_pos_cmd  (3)  absolute target position, world frame
        ee_quat_cmd (4)  absolute target quaternion wxyz, world frame
        gripper_cmd (1)  normalised [0=closed, 1=open]
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: dict, render_mode: Optional[str] = None):
        super().__init__()
        self.sim         = Simulation(config)
        self.render_mode = render_mode
        self._renderer   = None
        self._step_count = 0

        if render_mode == 'human':
            self._renderer = make_renderer(self.sim, config.get('rendering', {}))

        obs_low  = np.full(22, -np.inf, dtype=np.float32)
        obs_high = np.full(22,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action: [pos(3), quat(4), gripper(1)]
        act_low  = np.array([-np.inf]*3 + [-1]*4 + [0],  dtype=np.float32)
        act_high = np.array([ np.inf]*3 + [ 1]*4 + [1],  dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset sim with randomised pick/place, return obs and info."""
        super().reset(seed=seed)
        if seed is not None:
            self.sim._rng = np.random.default_rng(seed)

        self.sim.reset()
        self._step_count = 0

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        """Apply action, step sim, return (obs, reward, terminated, truncated, info)."""
        pos_world  = action[:3]
        quat_world = action[3:7]
        quat_world = quat_world / (np.linalg.norm(quat_world) + 1e-8)
        gripper_cmd = float(action[7])

        # Convert gripper [0=closed, 1=open] to MuJoCo ctrl [0=closed, 255=open]
        grasp = int(np.clip(gripper_cmd, 0.0, 1.0) * 255)

        self.sim.step_world(pos_world, quat_world, grasp=grasp)
        self._step_count += 1

        obs        = self._get_obs()
        info       = self._get_info()
        reward     = self._compute_reward(info)
        terminated = info['success']
        truncated  = self._step_count >= MAX_STEPS

        if self._renderer is not None:
            self._renderer.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is not None:
            self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

    def _get_obs(self):
        raw = self.sim.get_obs()
        parts = [ raw['ee_pos'], raw['ee_quat'], raw['obj_pos'], raw['obj_quat'], raw['pick_pos'], raw['place_pos'], [raw['gripper_width']], [0.0], ]
        obs = np.concatenate(parts).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        """Compute auxiliary info including success flag."""
        raw  = self.sim.get_obs()
        dist = np.linalg.norm(raw['obj_pos'] - raw['place_pos'])
        return {
            'obj_pos':       raw['obj_pos'],
            'dist_to_place': dist,
            'grasped':       raw['grasped'],
            'success':       dist < PLACE_THRESHOLD,
            'step':          self._step_count,
        }

    def _compute_reward(self, info: dict) -> float:
        """Dense reward: negative distance to place target, bonus on success."""
        reward = -info['dist_to_place']
        if info['success']:
            reward += 10.0
        return float(reward)
