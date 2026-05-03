import h5py, numpy as np
from src.data.loader import load_from_hdf5
from src.data.dataset import BCDataset

eps = load_from_hdf5('data/processed/dataset.h5')
ds  = BCDataset(eps[:5], normalise=False, subsample=10)
print("act_mean:", ds.act_mean.numpy())
print("act_std: ", ds.act_std.numpy())
print("delta_pos max:", ds.acts[:, :3].abs().max(0).values.numpy())