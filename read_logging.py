import h5py, numpy as np

with h5py.File('logs/rollouts/rollout_20260503_134917.h5', 'r') as f:
    ep = f['episode_0000']
    print("success:", ep.attrs['success'])
    print("steps:", ep.attrs['n_steps'])
    print("ee_pos z range:", ep['ee_pos'][:, 2].min(), ep['ee_pos'][:, 2].max())
    print("action z range:", ep['action'][:, 2].min(), ep['action'][:, 2].max())
    print("dist_to_place:", ep['dist_to_place'][:5])
    print("gripper_width:", ep['gripper_width'][:10])