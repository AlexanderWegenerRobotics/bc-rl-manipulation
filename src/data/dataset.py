import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from src.data.loader import Episode, load_from_hdf5


class BCDataset(Dataset):
    """
    PyTorch Dataset for behaviour cloning with delta actions.

    Observation vector per timestep (22-dim):
        ee_pos      (3)   world frame
        ee_quat     (4)   wxyz world frame
        obj_pos     (3)   world frame
        obj_quat    (4)   wxyz
        pick_pos    (3)   world frame
        place_pos   (3)   world frame
        gripper_w   (1)
        mode        (1)   0=unimanual, 1=bimanual

    Action vector per timestep (7-dim):
        delta_pos   (3)   world frame position delta
        delta_rot   (3)   world frame rotation vector delta
        gripper_cmd (1)   normalised [0=closed, 1=open]
    """

    OBS_DIM = 22
    ACT_DIM = 7

    def __init__(self, episodes: list[Episode], normalise: bool = True,
                 subsample: int = 1):
        self.subsample = subsample
        self.obs, self.acts = self._build_tensors(episodes)
        self.normalise = normalise

        if normalise:
            self.obs_mean = self.obs.mean(0)
            self.obs_std  = self.obs.std(0).clamp(min=1e-6)
            self.act_mean = self.acts.mean(0)
            self.act_std  = self.acts.std(0).clamp(min=1e-6)
        else:
            self.obs_mean = torch.zeros(self.OBS_DIM)
            self.obs_std  = torch.ones(self.OBS_DIM)
            self.act_mean = torch.zeros(self.ACT_DIM)
            self.act_std  = torch.ones(self.ACT_DIM)

    def _compute_delta_actions(self, ee_pos_cmd: np.ndarray,
                                ee_quat_cmd: np.ndarray) -> np.ndarray:
        """Compute delta pos and delta rotvec between consecutive commanded poses."""
        T = len(ee_pos_cmd)
        delta_pos = np.zeros((T, 3), dtype=np.float32)
        delta_rot = np.zeros((T, 3), dtype=np.float32)

        for t in range(T - 1):
            delta_pos[t] = ee_pos_cmd[t + 1] - ee_pos_cmd[t]
            r_curr = R.from_quat(ee_quat_cmd[t,  [1, 2, 3, 0]])
            r_next = R.from_quat(ee_quat_cmd[t+1,[1, 2, 3, 0]])
            r_delta = r_curr.inv() * r_next
            delta_rot[t] = r_delta.as_rotvec()

        # Last step gets zero delta
        delta_pos[-1] = 0.0
        delta_rot[-1] = 0.0
        return delta_pos, delta_rot

    def _build_tensors(self, episodes: list[Episode]) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten all episodes into (N, obs_dim) and (N, act_dim) tensors."""
        all_obs  = []
        all_acts = []

        for ep in episodes:
            # Subsample episode
            idx = np.arange(0, len(ep.ee_pos), self.subsample)
            ee_pos      = ep.ee_pos[idx]
            ee_quat     = ep.ee_quat[idx]
            ee_pos_cmd  = ep.ee_pos_cmd[idx]
            ee_quat_cmd = ep.ee_quat_cmd[idx]
            obj_pos     = ep.obj_pos[idx]
            obj_quat    = ep.obj_quat[idx]
            gripper_w   = ep.gripper_width[idx]
            T = len(ee_pos)

            pick_tiled  = np.tile(ep.pick_pos,  (T, 1))
            place_tiled = np.tile(ep.place_pos, (T, 1))
            mode_tiled  = np.full((T, 1), ep.mode, dtype=np.float32)

            gripper_w_col  = gripper_w.reshape(-1, 1)
            gripper_cmd    = np.clip(gripper_w / 0.08, 0.0, 1.0).reshape(-1, 1)

            delta_pos, delta_rot = self._compute_delta_actions(ee_pos_cmd, ee_quat_cmd)

            obs = np.concatenate([
                ee_pos, ee_quat,
                obj_pos, obj_quat,
                pick_tiled, place_tiled,
                gripper_w_col, mode_tiled,
            ], axis=1).astype(np.float32)

            act = np.concatenate([
                delta_pos, delta_rot, gripper_cmd,
            ], axis=1).astype(np.float32)

            all_obs.append(obs)
            all_acts.append(act)

        obs_tensor  = torch.from_numpy(np.concatenate(all_obs,  axis=0))
        acts_tensor = torch.from_numpy(np.concatenate(all_acts, axis=0))
        return obs_tensor, acts_tensor

    def normalise_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise an observation tensor using dataset statistics."""
        return (obs - self.obs_mean) / self.obs_std

    def denormalise_act(self, act: torch.Tensor) -> torch.Tensor:
        """Invert action normalisation."""
        return act * self.act_std + self.act_mean

    def get_stats(self) -> dict:
        """Return normalisation statistics as a plain dict for saving."""
        return {
            'obs_mean': self.obs_mean.numpy(),
            'obs_std':  self.obs_std.numpy(),
            'act_mean': self.act_mean.numpy(),
            'act_std':  self.act_std.numpy(),
        }

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.obs[idx]
        act = self.acts[idx]
        if self.normalise:
            obs = self.normalise_obs(obs)
            act = (act - self.act_mean) / self.act_std
        return obs, act


def make_datasets(hdf5_path: Path, val_split: float = 0.1,
                  normalise: bool = True, seed: int = 42,
                  subsample: int = 1) -> tuple[BCDataset, BCDataset]:
    """Load HDF5, split into train/val BCDatasets, fit normalisation on train only."""
    episodes = load_from_hdf5(hdf5_path)

    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(episodes))
    n_val   = max(1, int(len(episodes) * val_split))
    val_idx = indices[:n_val]
    trn_idx = indices[n_val:]

    train_eps = [episodes[i] for i in trn_idx]
    val_eps   = [episodes[i] for i in val_idx]

    train_ds = BCDataset(train_eps, normalise=normalise, subsample=subsample)
    val_ds   = BCDataset(val_eps,   normalise=False,     subsample=subsample)

    if normalise:
        val_ds.obs_mean  = train_ds.obs_mean
        val_ds.obs_std   = train_ds.obs_std
        val_ds.act_mean  = train_ds.act_mean
        val_ds.act_std   = train_ds.act_std
        val_ds.normalise = True

    print(f"Train: {len(train_ds)} steps from {len(train_eps)} episodes")
    print(f"Val:   {len(val_ds)}   steps from {len(val_eps)}   episodes")
    return train_ds, val_ds