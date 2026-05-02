import numpy as np
from scipy.spatial.transform import Rotation as R


class Pose:
    def __init__(self, position=None, quaternion=None):
        """
        Args:
            position: [x, y, z]
            quaternion: [w, x, y, z]
        """
        self.position = np.array(position if position is not None else [0, 0, 0], dtype=float)
        self.quaternion = np.array(quaternion if quaternion is not None else [1, 0, 0, 0], dtype=float)
        self.quaternion /= np.linalg.norm(self.quaternion)

    # ---- Factory methods ----

    @classmethod
    def from_matrix(cls, position, rotation_matrix):
        q_xyzw = R.from_matrix(rotation_matrix).as_quat()
        return cls(position, [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    @classmethod
    def from_rotvec(cls, position, rotvec):
        q_xyzw = R.from_rotvec(rotvec).as_quat()
        return cls(position, [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    @classmethod
    def from_7d(cls, vec):
        """From [px, py, pz, qw, qx, qy, qz]"""
        return cls(vec[:3], vec[3:])

    # ---- Getters ----

    @property
    def rotation_matrix(self):
        return R.from_quat(self._to_xyzw()).as_matrix()

    @property
    def rotvec(self):
        return R.from_quat(self._to_xyzw()).as_rotvec()

    @property
    def euler_rpy(self):
        return R.from_quat(self._to_xyzw()).as_euler('xyz')

    def as_7d(self):
        """Returns [px, py, pz, qw, qx, qy, qz]"""
        return np.concatenate([self.position, self.quaternion])

    # ---- Setters ----

    def set_position(self, position):
        self.position = np.array(position, dtype=float)

    def set_quaternion(self, quaternion):
        """quaternion: [w, x, y, z]"""
        self.quaternion = np.array(quaternion, dtype=float)
        self.quaternion /= np.linalg.norm(self.quaternion)

    def set_rotation_matrix(self, matrix):
        q_xyzw = R.from_matrix(matrix).as_quat()
        self.quaternion = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    def set_rotvec(self, rotvec):
        q_xyzw = R.from_rotvec(rotvec).as_quat()
        self.quaternion = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    # ---- Internal ----

    def _to_xyzw(self):
        """[w,x,y,z] -> [x,y,z,w] for scipy"""
        return self.quaternion[[1, 2, 3, 0]]

    def __repr__(self):
        return f"Pose(pos={self.position}, quat={self.quaternion})"