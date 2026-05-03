import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from scipy.spatial.transform import Rotation as R

from src.simulation.sim import Simulation
from src.simulation.rendering import make_renderer
from src.robot.pose import Pose


PLACE_THRESHOLD = 0.05
MAX_STEPS       = 1000


class PickPlaceEnv(gym.Env):
    """
    Gymnasium environment for unimanual pick and place with delta actions.

    Observation (22-dim):
        ee_pos      (3)  world frame
        ee_quat     (4)  wxyz world frame
        obj_pos     (3)  world frame
        obj_quat    (4)  wxyz
        pick_pos    (3)  world frame
        place_pos   (3)  world frame
        gripper_w   (1)
        mode        (1)  always 0 for unimanual

    Action (7-dim):
        delta_pos   (3)  position delta in world frame
        delta_rot   (3)  rotation vector delta in world frame
        gripper_cmd (1)  normalised [0=closed, 1=open]
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: dict, render_mode: Optional[str] = None):
        super().__init__()
        self.sim         = Simulation(config)
        self.render_mode = render_mode
        self._renderer   = None
        self._step_count = 0

        policy_cfg = config.get('policy', {})
        self._delta_pos_clip = policy_cfg.get('delta_pos_clip', 0.05)
        self._delta_rot_clip = policy_cfg.get('delta_rot_clip', 0.1)

        if render_mode == 'human':
            self._renderer = make_renderer(self.sim, config.get('rendering', {}))

        obs_low  = np.full(22, -np.inf, dtype=np.float32)
        obs_high = np.full(22,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        act_low  = np.array([-self._delta_pos_clip]*3 + [-self._delta_rot_clip]*3 + [0.0], dtype=np.float32)
        act_high = np.array([ self._delta_pos_clip]*3 + [ self._delta_rot_clip]*3 + [1.0], dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        self._current_pos_world  = None
        self._current_quat_world = None

    def reset(self, seed=None, options=None):
        """Reset sim with randomised pick/place and initialise current EE pose."""
        super().reset(seed=seed)
        if seed is not None:
            self.sim._rng = np.random.default_rng(seed)

        self.sim.reset()
        self._step_count = 0

        raw = self.sim.get_obs()
        self._current_pos_world  = raw['ee_pos'].copy()
        self._current_quat_world = raw['ee_quat'].copy()

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        pos_world   = action[:3]
        gripper_cmd = float(action[3])
        # use current EE orientation unchanged
        quat_world  = self._current_quat_world
        grasp = int(np.clip(gripper_cmd, 0.0, 1.0) * 255)
        self.sim.step_world(pos_world, quat_world, grasp=grasp)

        raw = self.sim.get_obs()
        self._current_pos_world  = raw['ee_pos'].copy()
        self._current_quat_world = raw['ee_quat'].copy()
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
        rel_to_obj   = raw['ee_pos'] - raw['obj_pos']
        rel_to_place = raw['ee_pos'] - raw['place_pos']
        obs = np.concatenate([
            raw['ee_pos'], raw['ee_quat'],
            raw['obj_pos'], raw['obj_quat'],
            raw['pick_pos'], raw['place_pos'],
            rel_to_obj, rel_to_place,
            [raw['gripper_width']], [0.0],
        ]).astype(np.float32)
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