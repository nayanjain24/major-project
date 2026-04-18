from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from app.ml.diffusion.noise import extract, linear_beta_schedule


class DenoiseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = t.float().unsqueeze(1) / 1000.0
        x_in = torch.cat([x, t_embed], dim=1)
        return self.net(x_in)


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    hidden_dim: int = 256
    device: str = "cpu"


class TabDDPM(nn.Module):
    def __init__(self, input_dim: int, config: DiffusionConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.device = torch.device(config.device)

        self.betas = linear_beta_schedule(config.timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]]
        )

        self.denoiser = DenoiseMLP(input_dim, config.hidden_dim).to(self.device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alphas_cumprod = torch.sqrt(extract(self.alphas_cumprod, t, x_start.shape))
        sqrt_one_minus = torch.sqrt(1.0 - extract(self.alphas_cumprod, t, x_start.shape))
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus * noise

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus = torch.sqrt(1.0 - extract(self.alphas_cumprod, t, x.shape))
        sqrt_recip_alphas = torch.sqrt(1.0 / extract(self.alphas, t, x.shape))
        model_mean = sqrt_recip_alphas * (x - betas_t * self.denoiser(x, t) / sqrt_one_minus)

        if (t == 0).all():
            return model_mean

        noise = torch.randn_like(x)
        posterior_var = betas_t * (1.0 - extract(self.alphas_cumprod_prev, t, x.shape)) / (
            1.0 - extract(self.alphas_cumprod, t, x.shape)
        )
        return model_mean + torch.sqrt(posterior_var) * noise

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x, t)
