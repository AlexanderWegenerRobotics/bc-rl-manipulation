import torch
from src.policy.bc_policy import BCPolicy

policy = BCPolicy.load('models/checkpoints/bc_policy.pt', device='cpu')
print("obs_mean ee_pos (indices 0-2):", policy.obs_mean[:3])
print("obs_std  ee_pos (indices 0-2):", policy.obs_std[:3])
print("act_mean ee_pos_cmd (indices 0-2):", policy.act_mean[:3])
print("act_std  ee_pos_cmd (indices 0-2):", policy.act_std[:3])
print("act_mean z (index 2):", policy.act_mean[2].item())
print("act_std  z (index 2):", policy.act_std[2].item())
