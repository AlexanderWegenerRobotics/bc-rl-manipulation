import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import make_datasets, BCDataset
from src.policy.bc_policy import BCPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BC policy from an HDF5 dataset.")
    parser.add_argument('--data',       type=Path,  default=Path('data/processed/dataset.h5'))
    parser.add_argument('--out',        type=Path,  default=Path('models/checkpoints/bc_policy.pt'))
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch-size', type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout',      type=float, default=0.1)
    parser.add_argument('--hidden',       type=int,   nargs='+', default=[256, 256, 128])
    parser.add_argument('--val-split',    type=float, default=0.15)
    parser.add_argument('--subsample',    type=int,   default=10,  help='Subsample every Nth step from each episode')
    parser.add_argument('--device',       type=str,   default='cpu')
    parser.add_argument('--seed',         type=int,   default=42)
    return parser.parse_args()


def train_epoch(policy: BCPolicy, loader: DataLoader,
                optimiser: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    """Run one training epoch, return mean loss."""
    policy.model.train()
    total_loss = 0.0
    for obs, act in loader:
        obs = obs.to(policy.device)
        act = act.to(policy.device)
        pred = policy.model(obs)
        loss = loss_fn(pred, act)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(obs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(policy: BCPolicy, loader: DataLoader, loss_fn: nn.Module) -> float:
    """Run one validation epoch, return mean loss."""
    policy.model.eval()
    total_loss = 0.0
    for obs, act in loader:
        obs  = obs.to(policy.device)
        act  = act.to(policy.device)
        pred = policy.model(obs)
        total_loss += loss_fn(pred, act).item() * len(obs)
    return total_loss / len(loader.dataset)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset from {args.data}")
    train_ds, val_ds = make_datasets(args.data, val_split=args.val_split,
                                     normalise=True, seed=args.seed,
                                     subsample=args.subsample)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    policy = BCPolicy(
        obs_dim     = BCDataset.OBS_DIM,
        act_dim     = BCDataset.ACT_DIM,
        hidden_dims = args.hidden,
        device      = args.device,
    )
    policy.set_normalisation(train_ds.get_stats())

    total_params = sum(p.numel() for p in policy.model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Training on {args.device} for {args.epochs} epochs\n")

    optimiser  = torch.optim.Adam(policy.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn    = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(policy, train_loader, optimiser, loss_fn)
        val_loss   = val_epoch(policy, val_loader, loss_fn)
        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            policy.save(args.out, extra={'epoch': epoch, 'val_loss': val_loss})

        if epoch % 10 == 0 or epoch == 1:
            marker = " *" if improved else ""
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"train {train_loss:.6f} | "
                  f"val {val_loss:.6f}{marker}")

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to {args.out}")


if __name__ == '__main__':
    main()