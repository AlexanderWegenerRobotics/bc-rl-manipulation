import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import mujoco
import threading
from dataclasses import dataclass
from typing import Optional
from scipy.spatial.transform import Rotation as R

from src.robot.robot_kinematics import RobotKinematics
from src.robot.pose import Pose
from src.robot.control import ImpedanceController


ARM_DOF = 7

PICK_X_RANGE  = (0.55, 0.90)
PICK_Y_RANGE  = (-0.35, 0.05)
PICK_Z        = 0.62

PLACE_X_RANGE = (0.65, 0.90)
PLACE_Y_RANGE = (-0.35, 0.05)
PLACE_Z       = 0.5 #0.435

MIN_PICK_PLACE_DIST = 0.10


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
        self._base_pos  = np.array(robot_cfg.get('base_pos',       [0.0, -0.4, 0.8]))
        self._base_quat = np.array(robot_cfg.get('base_quat_wxyz', [0.5, 0.5, 0.5, 0.5]))
        self._base_rot  = R.from_quat(self._base_quat[[1, 2, 3, 0]])

        self.kinematics = RobotKinematics(
            urdf_path      = robot_cfg['urdf_path'],
            ee_frame_name  = robot_cfg['ee_frame_name'],
            base_quat_wxyz = self._base_quat,
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

        ee_pose_base  = self.kinematics.forward_kinematics(q)
        ee_pos_world, ee_quat_world = self._base_to_world(
            ee_pose_base.position, ee_pose_base.quaternion)
        obj_pos, obj_quat = self._get_object_pose('box')
        gripper_width = finger_l + finger_r
        grasped       = contact and (self._grasp_width_min <= gripper_width <= self._grasp_width_max)

        return {
            'q':             q,
            'qd':            qd,
            'tau':           tau,
            'ee_pos':        ee_pos_world,
            'ee_quat':       ee_quat_world,
            'ee_pos_base':   ee_pose_base.position,
            'ee_quat_base':  ee_pose_base.quaternion,
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

    def _base_to_world(self, pos_base: np.ndarray, quat_base_wxyz: np.ndarray) -> tuple:
        """Transform position and quaternion from robot base frame to world frame."""
        pos_world  = self._base_rot.apply(pos_base) + self._base_pos
        rot_ee     = R.from_quat(quat_base_wxyz[[1, 2, 3, 0]])
        rot_world  = self._base_rot * rot_ee
        q_xyzw     = rot_world.as_quat()
        quat_world = q_xyzw[[3, 0, 1, 2]]
        return pos_world, quat_world

    def _world_to_base(self, pos_world: np.ndarray, quat_world_wxyz: np.ndarray) -> Pose:
        """Transform position and quaternion from world frame to robot base frame as Pose."""
        pos_base  = self._base_rot.inv().apply(pos_world - self._base_pos)
        rot_world = R.from_quat(quat_world_wxyz[[1, 2, 3, 0]])
        rot_base  = self._base_rot.inv() * rot_world
        q_xyzw    = rot_base.as_quat()
        quat_base = q_xyzw[[3, 0, 1, 2]]
        return Pose(position=pos_base, quaternion=quat_base)

    def step_world(self, pos_world: np.ndarray, quat_world_wxyz: np.ndarray, grasp: int = 255):
        """Step the sim given a target pose expressed in world frame."""
        target_base = self._world_to_base(pos_world, quat_world_wxyz)
        self.step(target_base, grasp)

    def _sample_pick_place(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample randomised pick and place positions respecting minimum distance."""
        for _ in range(100):
            px = self._rng.uniform(*PICK_X_RANGE)
            py = self._rng.uniform(*PICK_Y_RANGE)
            gx = self._rng.uniform(*PLACE_X_RANGE)
            gy = self._rng.uniform(*PLACE_Y_RANGE)
            if np.hypot(gx - px, gy - py) >= MIN_PICK_PLACE_DIST:
                return (np.array([px, py, PICK_Z]), np.array([gx, gy, PLACE_Z]))
        return (np.array([0.55, 0.15, PICK_Z]), np.array([0.55, -0.15, PLACE_Z]))

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
    import time
    from src.simulation.rendering import make_renderer

    with open('config/sim_config.yaml') as f:
        config = yaml.safe_load(f)

    sim      = Simulation(config)
    renderer = make_renderer(sim, config.get('rendering', {}))

    obs = sim.get_obs()
    print(f"Home ee_pos (world): {obs['ee_pos']}")

    # Find a q0 within the training z range (~0.65-0.70 world frame)
    q_candidates = [
        np.array([-0.3, -0.5,  0.3, -1.5,  0.5, 2.0, -0.9]),
        np.array([-0.3, -0.7,  0.3, -1.8,  0.5, 2.2, -0.9]),
        np.array([-0.3, -0.9,  0.3, -2.0,  0.5, 2.5, -0.9]),
        np.array([-0.3, -1.1,  0.3, -2.0,  0.5, 2.9, -0.9]),  # current home
    ]
    print("\nTesting candidate q0 poses:")
    for q in q_candidates:
        with sim._lock:
            sim.mj_data.qpos[:ARM_DOF] = q
            sim.mj_data.qvel[:] = 0.0
            mujoco.mj_forward(sim.mj_model, sim.mj_data)
        obs = sim.get_obs()
        print(f"  q={np.round(q,1)} -> ee_world={np.round(obs['ee_pos'],3)}")

    # Restore home pose
    sim.reset()

    obs = sim.get_obs()
    print(f"Initial EE pos (world): {obs['ee_pos']}")
    print(f"Initial EE pos (base):  {obs['ee_pos_base']}")

    # Controller tracking test:
    # Define a sequence of world-frame waypoints and verify the arm reaches each one.
    # Confirms base<->world transform and impedance control are correct
    # before attributing tracking failures to the policy.
    waypoints_world = [
        (np.array([0.55,  0.10, 0.75]), obs['ee_quat']),
        (np.array([0.55,  0.10, 0.62]), obs['ee_quat']),
        (np.array([0.55, -0.10, 0.62]), obs['ee_quat']),
        (np.array([0.55, -0.10, 0.75]), obs['ee_quat']),
    ]

    TRACKING_STEPS  = 400
    TRACKING_THRESH = 0.02

    print("\n--- Controller tracking test ---")
    for i, (target_pos, target_quat) in enumerate(waypoints_world):
        for _ in range(TRACKING_STEPS):
            sim.step_world(target_pos, target_quat, grasp=255)
            renderer.render()
            if renderer.stop_request:
                break

        obs = sim.get_obs()
        err = np.linalg.norm(obs['ee_pos'] - target_pos)
        status = "OK" if err < TRACKING_THRESH else "FAIL"
        print(f"  Waypoint {i+1}: target={np.round(target_pos,3)} "
              f"actual={np.round(obs['ee_pos'],3)} err={err:.4f}m [{status}]")

        if renderer.stop_request:
            break

    print("\nTracking test done. Holding final pose — press Q to quit.")
    while not renderer.stop_request:
        sim.step_world(waypoints_world[-1][0], waypoints_world[-1][1], grasp=255)
        renderer.render()

    renderer.close()