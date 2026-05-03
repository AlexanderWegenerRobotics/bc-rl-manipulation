import numpy as np
import h5py
from pathlib import Path
from datetime import datetime


class EpisodeLogger:
    """Logs per-step rollout data to HDF5. One group per episode."""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._path = self.log_dir / f'rollout_{timestamp}.h5'
        self._file = h5py.File(self._path, 'w')
        self._ep   = 0
        self._buf  = {}
        print(f"[EpisodeLogger] Writing to {self._path}")

    def start_episode(self, pick_pos: np.ndarray, place_pos: np.ndarray):
        """Begin a new episode buffer."""
        if not self.enabled:
            return
        self._buf = {
            'time':          [],
            'ee_pos':        [],
            'ee_quat':       [],
            'obj_pos':       [],
            'obj_quat':      [],
            'gripper_width': [],
            'action':        [],
            'dist_to_place': [],
        }
        self._pick_pos  = pick_pos.copy()
        self._place_pos = place_pos.copy()
        self._step      = 0

    def log_step(self, obs: dict, action: np.ndarray):
        """Append one step of obs + action to the buffer."""
        if not self.enabled:
            return
        dist = np.linalg.norm(obs['obj_pos'] - obs['place_pos'])
        self._buf['time'].append(float(self._step))
        self._buf['ee_pos'].append(obs['ee_pos'].copy())
        self._buf['ee_quat'].append(obs['ee_quat'].copy())
        self._buf['obj_pos'].append(obs['obj_pos'].copy())
        self._buf['obj_quat'].append(obs['obj_quat'].copy())
        self._buf['gripper_width'].append(float(obs['gripper_width']))
        self._buf['action'].append(action.copy())
        self._buf['dist_to_place'].append(float(dist))
        self._step += 1

    def end_episode(self, success: bool):
        """Flush buffer to HDF5 and start a new group."""
        if not self.enabled or not self._buf:
            return
        grp = self._file.create_group(f'episode_{self._ep:04d}')
        for key, val in self._buf.items():
            grp.create_dataset(key, data=np.array(val), compression='gzip')
        grp.attrs['pick_pos']  = self._pick_pos
        grp.attrs['place_pos'] = self._place_pos
        grp.attrs['success']   = success
        grp.attrs['n_steps']   = self._step
        self._file.flush()
        self._ep += 1
        self._buf = {}

    def close(self):
        """Flush and close the HDF5 file."""
        if not self.enabled:
            return
        self._file.close()
        print(f"[EpisodeLogger] Closed {self._path} ({self._ep} episodes)")
