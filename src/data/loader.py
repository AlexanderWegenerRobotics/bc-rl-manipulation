import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass


LEFT_BASE_POS  = np.array([0.0,  0.4, 0.8])
LEFT_BASE_QUAT = np.array([0.5, -0.5, 0.5, -0.5])   # wxyz
RIGHT_BASE_POS  = np.array([0.0, -0.4, 0.8])
RIGHT_BASE_QUAT = np.array([0.5,  0.5, 0.5,  0.5])  # wxyz

ENGAGED_STATE = 4
PLACE_THRESHOLD = 0.06


@dataclass
class Episode:
    ee_pos:        np.ndarray   # (T, 3) world frame
    ee_quat:       np.ndarray   # (T, 4) wxyz world frame
    ee_pos_cmd:    np.ndarray   # (T, 3) world frame
    ee_quat_cmd:   np.ndarray   # (T, 4) wxyz world frame
    gripper_width: np.ndarray   # (T,)
    obj_pos:       np.ndarray   # (T, 3) world frame
    obj_quat:      np.ndarray   # (T, 4) wxyz
    pick_pos:      np.ndarray   # (3,)
    place_pos:     np.ndarray   # (3,)
    mode:          int
    episode_id:    str


def _base_rot(quat_wxyz: np.ndarray) -> R:
    """Build scipy Rotation from a wxyz quaternion."""
    q = quat_wxyz
    return R.from_quat([q[1], q[2], q[3], q[0]])


def _transform_ee_to_world(pos_base: np.ndarray, quat_base: np.ndarray,
                            base_pos: np.ndarray, base_quat_wxyz: np.ndarray):
    """Transform EE positions and quaternions from robot base frame to world frame."""
    rot_base = _base_rot(base_quat_wxyz)

    pos_world = rot_base.apply(pos_base) + base_pos

    rot_ee = R.from_quat(quat_base[:, [1, 2, 3, 0]])
    rot_world = rot_base * rot_ee
    quat_xyzw = rot_world.as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    return pos_world, quat_wxyz


def _extract_ee_from_arm(df: pd.DataFrame) -> tuple:
    """Extract EE position and quaternion from flat O_T_EE columns (column-major 4x4)."""
    pos  = df[['O_T_EE_12', 'O_T_EE_13', 'O_T_EE_14']].values
    rot_flat = df[[f'O_T_EE_{i}' for i in range(12)]].values.reshape(-1, 3, 4)[:, :, :3]
    quat_list = []
    for mat in rot_flat:
        q_xyzw = R.from_matrix(mat).as_quat()
        quat_list.append([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    quat = np.array(quat_list)
    return pos, quat


def _extract_ee_cmd_from_arm(df: pd.DataFrame) -> tuple:
    """Extract commanded EE position and quaternion from O_T_EE_cmd columns."""
    pos  = df[['O_T_EE_cmd_12', 'O_T_EE_cmd_13', 'O_T_EE_cmd_14']].values
    rot_flat = df[[f'O_T_EE_cmd_{i}' for i in range(12)]].values.reshape(-1, 3, 4)[:, :, :3]
    quat_list = []
    for mat in rot_flat:
        q_xyzw = R.from_matrix(mat).as_quat()
        quat_list.append([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    quat = np.array(quat_list)
    return pos, quat


def _load_meta(meta_path: Path) -> dict:
    """Load episode config (pick, place, mode) from meta CSV."""
    meta = pd.read_csv(meta_path, sep=';')
    cfg  = meta[meta['event'] == 'episode_config']
    if cfg.empty:
        return None
    row = cfg.iloc[0]
    return {
        'pick_pos':  np.array([row['pick_x'],  row['pick_y'],  row['pick_z']]),
        'place_pos': np.array([row['place_x'], row['place_y'], row['place_z']]),
        'mode':      int(row['mode']),
    }


def _filter_engaged(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where state == ENGAGED."""
    return df[df['state'] == ENGAGED_STATE].copy()


def _trim_after_success(arm_df: pd.DataFrame, scene_df: pd.DataFrame,
                         place_pos: np.ndarray) -> tuple:
    """Trim timesteps after the object first reaches the place target."""
    obj_pos   = scene_df[['object_x', 'object_y', 'object_z']].values
    dist      = np.linalg.norm(obj_pos - place_pos, axis=1)
    success   = np.where(dist < PLACE_THRESHOLD)[0]

    if len(success) == 0:
        return arm_df, scene_df

    cut_time = scene_df['time'].iloc[success[0]]
    arm_df   = arm_df[arm_df['time'] <= cut_time]
    scene_df = scene_df[scene_df['time'] <= cut_time]
    return arm_df, scene_df


def _align_by_time(arm_df: pd.DataFrame, scene_df: pd.DataFrame) -> pd.DataFrame:
    """Merge arm and scene dataframes on nearest timestamp."""
    arm_df   = arm_df.sort_values('time').reset_index(drop=True)
    scene_df = scene_df.sort_values('time').reset_index(drop=True)
    merged   = pd.merge_asof(arm_df, scene_df, on='time', direction='nearest',
                              suffixes=('_arm', '_scene'))
    return merged


def load_episode(episode_dir: Path, arm: str = 'arm_left') -> Episode | None:
    """Load, transform, and filter one episode folder into an Episode dataclass."""
    arm_path   = episode_dir / f'{arm}.csv'
    scene_path = episode_dir / 'scene.csv'
    meta_path  = episode_dir / f'{arm}_meta.csv'

    for p in [arm_path, scene_path, meta_path]:
        if not p.exists():
            print(f"[SKIP] Missing {p}")
            return None

    meta = _load_meta(meta_path)
    if meta is None:
        print(f"[SKIP] No episode config in {meta_path}")
        return None

    arm_df   = pd.read_csv(arm_path,   sep=';')
    scene_df = pd.read_csv(scene_path, sep=';')

    if arm_df.empty or scene_df.empty:
        print(f"[SKIP] Empty CSV in {episode_dir}")
        return None

    arm_df   = _filter_engaged(arm_df)
    scene_df = _filter_engaged(scene_df) if 'state' in scene_df.columns else scene_df

    if arm_df.empty:
        print(f"[SKIP] No engaged rows in {episode_dir}")
        return None

    arm_df, scene_df = _trim_after_success(arm_df, scene_df, meta['place_pos'])

    merged = _align_by_time(arm_df, scene_df)

    base_pos  = LEFT_BASE_POS  if 'left' in arm else RIGHT_BASE_POS
    base_quat = LEFT_BASE_QUAT if 'left' in arm else RIGHT_BASE_QUAT

    pos_base, quat_base         = _extract_ee_from_arm(merged)
    pos_cmd_base, quat_cmd_base = _extract_ee_cmd_from_arm(merged)

    ee_pos,     ee_quat     = _transform_ee_to_world(pos_base,     quat_base,     base_pos, base_quat)
    ee_pos_cmd, ee_quat_cmd = _transform_ee_to_world(pos_cmd_base, quat_cmd_base, base_pos, base_quat)

    obj_pos  = merged[['object_x', 'object_y', 'object_z']].values
    obj_quat = merged[['object_qw', 'object_qx', 'object_qy', 'object_qz']].values

    gripper_width = merged['gripper_width'].values

    return Episode(
        ee_pos        = ee_pos,
        ee_quat       = ee_quat,
        ee_pos_cmd    = ee_pos_cmd,
        ee_quat_cmd   = ee_quat_cmd,
        gripper_width = gripper_width,
        obj_pos       = obj_pos,
        obj_quat      = obj_quat,
        pick_pos      = meta['pick_pos'],
        place_pos     = meta['place_pos'],
        mode          = meta['mode'],
        episode_id    = episode_dir.name,
    )


def load_all_episodes(log_root: Path, arm: str = 'arm_left') -> list[Episode]:
    """Load all episode folders from log_root, skipping incomplete or failed ones."""
    dirs     = sorted([d for d in log_root.iterdir() if d.is_dir()])
    episodes = []
    for d in dirs:
        ep = load_episode(d, arm=arm)
        if ep is not None:
            episodes.append(ep)
    print(f"Loaded {len(episodes)} / {len(dirs)} episodes.")
    return episodes


def save_to_hdf5(episodes: list[Episode], out_path: Path):
    """Save list of Episode objects to a single HDF5 file, one group per episode."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        for ep in episodes:
            grp = f.create_group(ep.episode_id)
            grp.create_dataset('ee_pos',        data=ep.ee_pos,        compression='gzip')
            grp.create_dataset('ee_quat',       data=ep.ee_quat,       compression='gzip')
            grp.create_dataset('ee_pos_cmd',    data=ep.ee_pos_cmd,    compression='gzip')
            grp.create_dataset('ee_quat_cmd',   data=ep.ee_quat_cmd,   compression='gzip')
            grp.create_dataset('gripper_width', data=ep.gripper_width, compression='gzip')
            grp.create_dataset('obj_pos',       data=ep.obj_pos,       compression='gzip')
            grp.create_dataset('obj_quat',      data=ep.obj_quat,      compression='gzip')
            grp.attrs['pick_pos']  = ep.pick_pos
            grp.attrs['place_pos'] = ep.place_pos
            grp.attrs['mode']      = ep.mode
    print(f"Saved {len(episodes)} episodes to {out_path}")


def load_from_hdf5(path: Path) -> list[Episode]:
    """Load Episode objects from an HDF5 file produced by save_to_hdf5."""
    episodes = []
    with h5py.File(path, 'r') as f:
        for ep_id, grp in f.items():
            ep = Episode(
                ee_pos        = grp['ee_pos'][:],
                ee_quat       = grp['ee_quat'][:],
                ee_pos_cmd    = grp['ee_pos_cmd'][:],
                ee_quat_cmd   = grp['ee_quat_cmd'][:],
                gripper_width = grp['gripper_width'][:],
                obj_pos       = grp['obj_pos'][:],
                obj_quat      = grp['obj_quat'][:],
                pick_pos      = grp.attrs['pick_pos'],
                place_pos     = grp.attrs['place_pos'],
                mode          = int(grp.attrs['mode']),
                episode_id    = ep_id,
            )
            episodes.append(ep)
    return episodes