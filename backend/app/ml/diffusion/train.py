from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.diffusion.tabddpm import DiffusionConfig, TabDDPM


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3


def train_diffusion(
    data: torch.Tensor, config: DiffusionConfig, train_cfg: TrainConfig
) -> Tuple[TabDDPM, Dict[str, float]]:
    model = TabDDPM(input_dim=data.shape[1], config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(train_cfg.epochs):
        for (batch,) in loader:
            batch = batch.to(model.device)
            t = torch.randint(0, config.timesteps, (batch.shape[0],), device=model.device)
            noise = torch.randn_like(batch)
            x_t = model.q_sample(batch, t, noise)
            pred_noise = model(x_t, t)
            loss = loss_fn(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, {"final_loss": float(loss.item())}
