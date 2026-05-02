import numpy as np
import pinocchio as pin
from src.robot.pose import Pose

ARM_DOF = 7


class RobotKinematics:
    def __init__(self, urdf_path: str, ee_frame_name: str = "fr3_hand_tcp", q_hand_default=None):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv

        if q_hand_default is None:
            q_hand_default = np.zeros(self.nq - ARM_DOF)
        self.q_hand_default = np.array(q_hand_default, dtype=float)

        try:
            self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        except:
            print(f"Available frames: {[self.model.frames[i].name for i in range(self.model.nframes)]}")
            raise ValueError(f"Frame '{ee_frame_name}' not found in URDF")

    def _full_q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape[0] == self.nq:
            return q
        if q.shape[0] == ARM_DOF:
            return np.concatenate([q, self.q_hand_default])
        raise ValueError(f"Expected q of length {ARM_DOF} or {self.nq}, got {q.shape[0]}")

    def _full_qv(self, q, qd=None):
        q_full = self._full_q(q)
        if qd is None:
            return q_full
        qd = np.asarray(qd, dtype=float)
        if qd.shape[0] == self.nv:
            qd_full = qd
        elif qd.shape[0] == ARM_DOF:
            qd_full = np.concatenate([qd, np.zeros(self.nv - ARM_DOF)])
        else:
            raise ValueError(f"Expected qd of length {ARM_DOF} or {self.nv}, got {qd.shape[0]}")
        return q_full, qd_full

    def forward_kinematics(self, q):
        q_full = self._full_q(q)
        pin.framesForwardKinematics(self.model, self.data, q_full)
        pose = self.data.oMf[self.ee_frame_id]
        return Pose.from_matrix(pose.translation.copy(), pose.rotation.copy())

    def get_jacobian(self, q):
        q_full = self._full_q(q)
        pin.computeJointJacobians(self.model, self.data, q_full)
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J[:, :ARM_DOF]

    def get_ee_velocity(self, q, qd):
        J = self.get_jacobian(q)
        return J @ qd[:ARM_DOF]

    def get_gravity_torques(self, q):
        q_full = self._full_q(q)
        pin.forwardKinematics(self.model, self.data, q_full)
        tau_g = pin.computeGeneralizedGravity(self.model, self.data, q_full)
        return tau_g[:ARM_DOF]

    def get_mass_matrix(self, q):
        q_full = self._full_q(q)
        pin.crba(self.model, self.data, q_full)
        return self.data.M[:ARM_DOF, :ARM_DOF]

    def get_coriolis_matrix(self, q, qd):
        q_full, qd_full = self._full_qv(q, qd)
        pin.computeCoriolisMatrix(self.model, self.data, q_full, qd_full)
        return self.data.C[:ARM_DOF, :ARM_DOF]

    def get_internal_wrench(self, q, qd, tau_m):
        C = self.get_coriolis_matrix(q, qd)
        g = self.get_gravity_torques(q)
        q_full = self._full_q(q)
        pin.computeJointJacobians(self.model, self.data, q_full)
        J_body = pin.getFrameJacobian(
            self.model, self.data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL
        )
        J_body = J_body[:, :ARM_DOF]
        J_body_pinv = np.linalg.pinv(J_body)
        return J_body_pinv.T @ (tau_m - C @ qd[:ARM_DOF] - g)

    def get_cartesian_mass_matrix(self, q):
        M = self.get_mass_matrix(q)
        J = self.get_jacobian(q)
        M_inv = np.linalg.inv(M)
        J_M_inv_JT = J @ M_inv @ J.T
        return np.linalg.inv(J_M_inv_JT)