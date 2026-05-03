from src.data.dataset import make_datasets

train_ds, val_ds = make_datasets(
    'data/processed/dataset.h5',
    subsample=5, normalise=True, norm_acts=False
)
print("act abs mean:", train_ds.acts.abs().mean(0))
print("act abs max:", train_ds.acts.abs().max(0).values)