import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.spatial.transform import Rotation as R


class MLPPolicy(nn.Module):
    """MLP policy for behaviour cloning. Maps obs -> delta action in a single forward pass."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dims: list[int] = [256, 256, 128],
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class BCPolicy:
    """Wraps MLPPolicy with normalisation, device management, and save/load."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dims: list[int] = [256, 256, 128],
                 dropout: float = 0.0,
                 device: str = 'auto'):
        self.dropout  = dropout
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        print(f"Using {device} for training")
        self.device = torch.device(device)
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.model    = MLPPolicy(obs_dim, act_dim, hidden_dims, dropout).to(self.device)

        self.obs_mean = torch.zeros(obs_dim, device=self.device)
        self.obs_std  = torch.ones(obs_dim,  device=self.device)
        self.act_mean = torch.zeros(act_dim, device=self.device)
        self.act_std  = torch.ones(act_dim,  device=self.device)

    def set_normalisation(self, stats: dict):
        """Load normalisation statistics from dataset."""
        self.obs_mean = torch.tensor(stats['obs_mean'], dtype=torch.float32, device=self.device)
        self.obs_std  = torch.tensor(stats['obs_std'],  dtype=torch.float32, device=self.device)
        self.act_mean = torch.tensor(stats['act_mean'], dtype=torch.float32, device=self.device)
        self.act_std  = torch.tensor(stats['act_std'],  dtype=torch.float32, device=self.device)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run a single forward pass, returning a denormalised delta action as numpy."""
        self.model.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            obs_n = (obs_t - self.obs_mean) / self.obs_std
            act_n = self.model(obs_n)
            act   = act_n * self.act_std + self.act_mean
        return act.cpu().numpy()

    def save(self, path: Path, extra: dict = None):
        """Save model weights and normalisation stats to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'model_state': self.model.state_dict(),
            'obs_dim':     self.obs_dim,
            'act_dim':     self.act_dim,
            'obs_mean':    self.obs_mean.cpu().numpy(),
            'obs_std':     self.obs_std.cpu().numpy(),
            'act_mean':    self.act_mean.cpu().numpy(),
            'act_std':     self.act_std.cpu().numpy(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"Saved policy to {path}")

    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'BCPolicy':
        """Load a saved BCPolicy from disk."""
        payload = torch.load(path, map_location=device, weights_only=False)
        policy  = cls(obs_dim=payload['obs_dim'], act_dim=payload['act_dim'], device=device)
        policy.model.load_state_dict(payload['model_state'])
        policy.set_normalisation({
            'obs_mean': payload['obs_mean'],
            'obs_std':  payload['obs_std'],
            'act_mean': payload['act_mean'],
            'act_std':  payload['act_std'],
        })
        return policy