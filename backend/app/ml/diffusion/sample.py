from __future__ import annotations

import torch

from app.ml.diffusion.tabddpm import DiffusionConfig, TabDDPM


def sample_diffusion(model: TabDDPM, samples: int, config: DiffusionConfig) -> torch.Tensor:
    model.eval()
    x = torch.randn(samples, model.input_dim, device=model.device)
    with torch.no_grad():
        for t in reversed(range(config.timesteps)):
            t_batch = torch.full((samples,), t, device=model.device, dtype=torch.long)
            x = model.p_sample(x, t_batch)
    return x
