from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.services.preprocessing import Preprocessor


class ImputationService:
    def __init__(self) -> None:
        self.preprocessor = Preprocessor()

    def simple_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
            else:
                fill_value = df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else "__MISSING__"
            output[col] = df[col].fillna(fill_value)
        return output

    def diffusion_impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        # Lazy imports keep backend startup lightweight and avoid loading torch for non-ML routes.
        import torch

        from app.ml.diffusion.tabddpm import DiffusionConfig
        from app.ml.diffusion.train import TrainConfig, train_diffusion

        imputed = self.simple_impute(df)

        data_matrix, artifacts = self.preprocessor.fit_transform(imputed)
        tensor_data = torch.tensor(data_matrix)

        diff_cfg = DiffusionConfig(timesteps=200, hidden_dim=256, device="cpu")
        train_cfg = TrainConfig(epochs=3, batch_size=128, lr=1e-3)
        model, metrics = train_diffusion(tensor_data, diff_cfg, train_cfg)
        model.eval()

        # Placeholder reconstruction: in production replace with mask-aware sampling.
        reconstructed = self.preprocessor.inverse_transform(data_matrix, artifacts)
        reconstructed = reconstructed[imputed.columns]

        metrics.update({"imputed_cells": float(df.isna().sum().sum())})
        return reconstructed, metrics

    def impute(self, df: pd.DataFrame, target_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict[str, float]]:
        if target_columns:
            subset = df[target_columns]
            imputed_subset, metrics = self.diffusion_impute(subset)
            output = df.copy()
            output[target_columns] = imputed_subset
            return output, metrics
        return self.diffusion_impute(df)
