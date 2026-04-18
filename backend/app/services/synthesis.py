from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from app.services.preprocessing import Preprocessor


class SynthesisService:
    def __init__(self) -> None:
        self.preprocessor = Preprocessor()

    def generate(self, df: pd.DataFrame, samples: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
        # Lazy imports keep backend startup lightweight and avoid loading torch for non-ML routes.
        import torch

        from app.ml.diffusion.sample import sample_diffusion
        from app.ml.diffusion.tabddpm import DiffusionConfig
        from app.ml.diffusion.train import TrainConfig, train_diffusion

        processed, artifacts = self.preprocessor.fit_transform(df)
        tensor_data = torch.tensor(processed)

        diff_cfg = DiffusionConfig(timesteps=200, hidden_dim=256, device="cpu")
        train_cfg = TrainConfig(epochs=3, batch_size=128, lr=1e-3)
        model, metrics = train_diffusion(tensor_data, diff_cfg, train_cfg)

        generated = sample_diffusion(model, samples, diff_cfg)
        generated_df = self.preprocessor.inverse_transform(generated.cpu().numpy(), artifacts)
        generated_df = generated_df[df.columns]

        metrics.update({"samples": float(samples)})
        return generated_df, metrics
