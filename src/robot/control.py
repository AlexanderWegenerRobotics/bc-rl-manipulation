import numpy as np
import pinocchio as pin

from src.robot.robot_kinematics import RobotKinematics
from src.robot.pose import Pose


class ImpedanceController:
    def __init__(self, config: dict, robot_kinematics: RobotKinematics):
        self.robot_kin  = robot_kinematics
        self.K_cart     = np.diag(config['K_cart'])
        self.D_cart     = np.diag(config['D_cart'])
        self.K_null     = config['K_null']
        self.tau_max    = np.array(config['tau_max'])
        self.q_nominal  = np.array(config['q_nominal'])
        self.gravity_comp = config.get('gravity_compensation', True)

    def compute_control(self, q: np.ndarray, qd: np.ndarray, target: Pose) -> np.ndarray:
        """Compute joint torques to track a Cartesian target pose via impedance control."""
        x_current  = self.robot_kin.forward_kinematics(q)
        xd_current = self.robot_kin.get_ee_velocity(q, qd)
        J          = self.robot_kin.get_jacobian(q)

        e_pos = target.position - x_current.position
        R_err = target.rotation_matrix @ x_current.rotation_matrix.T
        e_rot = pin.log3(R_err)
        e     = np.concatenate([e_pos, e_rot])

        F        = self.K_cart @ e + self.D_cart @ (-xd_current)
        tau_task = J.T @ F

        J_pinv         = np.linalg.pinv(J)
        null_projector = np.eye(len(q)) - J_pinv @ J
        tau_null       = self.K_null * (self.q_nominal - q)

        tau = tau_task + null_projector @ tau_null

        if self.gravity_comp:
            tau += self.robot_kin.get_gravity_torques(q)

        return self._clip_torques(tau)

    def set_params(self, params: dict):
        """Update controller gains at runtime."""
        if 'K_cart'    in params: self.K_cart    = np.diag(params['K_cart'])
        if 'D_cart'    in params: self.D_cart    = np.diag(params['D_cart'])
        if 'K_null'    in params: self.K_null    = params['K_null']
        if 'q_nominal' in params: self.q_nominal = np.array(params['q_nominal'])

    def _clip_torques(self, tau: np.ndarray) -> np.ndarray:
        """Clip torques to hardware limits."""
        tau_clipped = np.clip(tau, -self.tau_max, self.tau_max)
        if not np.allclose(tau, tau_clipped):
            exceeded = np.where(np.abs(tau) > self.tau_max)[0]
            print(f"Warning: Torque limits exceeded on joints {exceeded}")
        return tau_clipped
