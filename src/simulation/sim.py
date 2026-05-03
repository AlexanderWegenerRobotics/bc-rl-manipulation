import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import mujoco
import threading
from dataclasses import dataclass
from typing import Optional

from src.robot.robot_kinematics import RobotKinematics
from src.robot.pose import Pose
from src.robot.control import ImpedanceController


ARM_DOF = 7

PICK_X_RANGE = (0.45, 0.70)
PICK_Y_RANGE = (-0.35, 0.35)
PICK_Z       = 0.62

PLACE_X_RANGE = (0.45, 0.70)
PLACE_Y_RANGE = (-0.35, 0.35)
PLACE_Z       = 0.62

MIN_PICK_PLACE_DIST = 0.15


@dataclass
class RobotState:
    q:       np.ndarray
    qd:      np.ndarray
    tau:     np.ndarray
    ee_pose: Pose


class Simulation:
    def __init__(self, config: dict):
        self.config    = config
        self.mj_model  = mujoco.MjModel.from_xml_path(config['simulation']['scene_path'])
        self.mj_data   = mujoco.MjData(self.mj_model)
        self._lock     = threading.Lock()

        self.steps_per_action = config['simulation']['steps_per_action']
        self.q0 = np.array(config['simulation']['q0'])

        robot_cfg       = config['robot']
        self.kinematics = RobotKinematics(
            urdf_path     = robot_cfg['urdf_path'],
            ee_frame_name = robot_cfg['ee_frame_name'],
            base_quat_wxyz = robot_cfg.get('base_quat_wxyz'),
        )
        self.controller = ImpedanceController(config['control'], self.kinematics)

        self._gripper_ctrl_idx = robot_cfg['gripper']['ctrl_index']
        self._finger_idx_left  = robot_cfg['finger_joints']['index_left']
        self._finger_idx_right = robot_cfg['finger_joints']['index_right']

        grasp_cfg = config['grasp_detection']
        self._left_finger_id  = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, grasp_cfg['contact_bodies'][0])
        self._right_finger_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, grasp_cfg['contact_bodies'][1])
        self._box_body_id     = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, grasp_cfg['object_body'])
        self._grasp_width_min = grasp_cfg['width_min']
        self._grasp_width_max = grasp_cfg['width_max']

        self._place_target_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'place_target')

        self._target_pose: Optional[Pose] = None
        self._pick_pos:  np.ndarray = np.array([0.6, 0.0, PICK_Z])
        self._place_pos: np.ndarray = np.array([0.6, 0.2, PLACE_Z])
        self._rng = np.random.default_rng()

        self.reset()

    def reset(self, pick_pos: Optional[np.ndarray] = None, place_pos: Optional[np.ndarray] = None):
        """Reset to q0 and randomise or apply given pick/place positions."""
        if pick_pos is None or place_pos is None:
            pick_pos, place_pos = self._sample_pick_place()

        self._pick_pos  = pick_pos
        self._place_pos = place_pos

        with self._lock:
            self.mj_data.qpos[:ARM_DOF] = self.q0
            self.mj_data.qvel[:]         = 0.0
            self.mj_data.qacc[:]         = 0.0
            self.mj_data.ctrl[:]         = 0.0
            self.mj_data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self._target_pose = self.kinematics.forward_kinematics(self.q0)

        self._set_object_pose('box', pick_pos, np.array([1, 0, 0, 0]))
        self._set_mocap_pos('place_target', place_pos)

    def step(self, target_pose: Pose, grasp: int = 255):
        """Advance physics by steps_per_action steps with impedance control."""
        self._target_pose = target_pose
        for _ in range(self.steps_per_action):
            with self._lock:
                q   = self.mj_data.qpos[:ARM_DOF].copy()
                qd  = self.mj_data.qvel[:ARM_DOF].copy()
                tau = self.controller.compute_control(q, qd, self._target_pose)
                self.mj_data.ctrl[:ARM_DOF]            = tau
                self.mj_data.ctrl[self._gripper_ctrl_idx] = grasp
                mujoco.mj_step(self.mj_model, self.mj_data)

    def get_obs(self) -> dict:
        """Return current observation dict in world frame."""
        with self._lock:
            q        = self.mj_data.qpos[:ARM_DOF].copy()
            qd       = self.mj_data.qvel[:ARM_DOF].copy()
            tau      = self.mj_data.ctrl[:ARM_DOF].copy()
            finger_l = self.mj_data.qpos[self._finger_idx_left]
            finger_r = self.mj_data.qpos[self._finger_idx_right]
            contact  = self._detect_contact()

        ee_pose       = self.kinematics.forward_kinematics(q)
        obj_pos, obj_quat = self._get_object_pose('box')
        gripper_width = finger_l + finger_r
        grasped       = contact and (self._grasp_width_min <= gripper_width <= self._grasp_width_max)

        return {
            'q':             q,
            'qd':            qd,
            'tau':           tau,
            'ee_pos':        ee_pose.position,
            'ee_quat':       ee_pose.quaternion,
            'obj_pos':       obj_pos,
            'obj_quat':      obj_quat,
            'gripper_width': gripper_width,
            'contact':       contact,
            'grasped':       grasped,
            'pick_pos':      self._pick_pos.copy(),
            'place_pos':     self._place_pos.copy(),
        }

    def get_state(self) -> RobotState:
        """Return current RobotState."""
        with self._lock:
            q   = self.mj_data.qpos[:ARM_DOF].copy()
            qd  = self.mj_data.qvel[:ARM_DOF].copy()
            tau = self.mj_data.ctrl[:ARM_DOF].copy()
        return RobotState(q=q, qd=qd, tau=tau, ee_pose=self.kinematics.forward_kinematics(q))

    @property
    def dt(self) -> float:
        return self.mj_model.opt.timestep * self.steps_per_action

    @property
    def pick_pos(self) -> np.ndarray:
        return self._pick_pos.copy()

    @property
    def place_pos(self) -> np.ndarray:
        return self._place_pos.copy()

    def _sample_pick_place(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample randomised pick and place positions respecting minimum distance."""
        for _ in range(100):
            px = self._rng.uniform(*PICK_X_RANGE)
            py = self._rng.uniform(*PICK_Y_RANGE)
            gx = self._rng.uniform(*PLACE_X_RANGE)
            gy = self._rng.uniform(*PLACE_Y_RANGE)
            if np.hypot(gx - px, gy - py) >= MIN_PICK_PLACE_DIST:
                return (np.array([px, py, PICK_Z]),
                        np.array([gx, gy, PLACE_Z]))
        return (np.array([0.55, 0.15, PICK_Z]),
                np.array([0.55, -0.15, PLACE_Z]))

    def _detect_contact(self) -> bool:
        """Return True if either finger is in contact with the box."""
        box_id = self._box_body_id
        for i in range(self.mj_data.ncon):
            c  = self.mj_data.contact[i]
            b1 = self.mj_model.geom_bodyid[c.geom1]
            b2 = self.mj_model.geom_bodyid[c.geom2]
            box_involved  = b1 == box_id or b2 == box_id
            left_touched  = b1 == self._left_finger_id  or b2 == self._left_finger_id
            right_touched = b1 == self._right_finger_id or b2 == self._right_finger_id
            if box_involved and (left_touched or right_touched):
                return True
        return False

    def _set_object_pose(self, name: str, pos: np.ndarray, quat: np.ndarray):
        """Set freejoint body pose and zero its velocity."""
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Body '{name}' not found")
        jnt_ids = [j for j in range(self.mj_model.njnt)
                   if self.mj_model.jnt_bodyid[j] == body_id]
        with self._lock:
            if jnt_ids and self.mj_model.jnt_type[jnt_ids[0]] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = self.mj_model.jnt_qposadr[jnt_ids[0]]
                dadr = self.mj_model.jnt_dofadr[jnt_ids[0]]
                self.mj_data.qpos[qadr:qadr+3] = pos
                self.mj_data.qpos[qadr+3:qadr+7] = quat
                self.mj_data.qvel[dadr:dadr+6] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)

    def _get_object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return world-space position and quaternion (wxyz) of a named body."""
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Body '{name}' not found")
        with self._lock:
            pos  = self.mj_data.xpos[body_id].copy()
            quat = self.mj_data.xquat[body_id].copy()
        return pos, quat

    def _set_mocap_pos(self, name: str, pos: np.ndarray):
        """Move a mocap body to the given world position."""
        body_id  = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        mocap_id = self.mj_model.body_mocapid[body_id]
        if mocap_id == -1:
            raise ValueError(f"Body '{name}' is not a mocap body")
        with self._lock:
            self.mj_data.mocap_pos[mocap_id] = pos


if __name__ == '__main__':
    import yaml
    from src.simulation.rendering import make_renderer

    with open('config/sim_config.yaml') as f:
        config = yaml.safe_load(f)

    sim = Simulation(config)
    renderer = make_renderer(sim, config.get('rendering', {}))

    print(f"Initial EE pos: {sim.get_obs()['ee_pos']}")
    print("Running — press Q to quit")

    while not renderer.stop_request:
        obs = sim.get_obs()
        target = Pose(position=obs['ee_pos'], quaternion=obs['ee_quat'])
        sim.step(target, grasp=255)
        renderer.render()

    renderer.close()