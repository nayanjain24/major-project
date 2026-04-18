import torch


def linear_beta_schedule(timesteps: int, start: float = 1e-4, end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(start, end, timesteps)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.gather(-1, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out
