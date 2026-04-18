from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessArtifacts:
    numeric_columns: List[str]
    categorical_columns: List[str]
    scaler: StandardScaler
    encoder: OneHotEncoder


class Preprocessor:
    def split_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [c for c in df.columns if c not in numeric_columns]
        return numeric_columns, categorical_columns

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, PreprocessArtifacts]:
        numeric_columns, categorical_columns = self.split_columns(df)

        numeric_data = df[numeric_columns].to_numpy(dtype=np.float32) if numeric_columns else np.empty((len(df), 0))
        scaler = StandardScaler()
        if numeric_columns:
            numeric_data = scaler.fit_transform(np.nan_to_num(numeric_data, nan=0.0))

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_data = df[categorical_columns].astype(str).fillna("__MISSING__") if categorical_columns else pd.DataFrame()
        if categorical_columns:
            cat_encoded = encoder.fit_transform(cat_data)
        else:
            cat_encoded = np.empty((len(df), 0))

        combined = np.concatenate([numeric_data, cat_encoded], axis=1)

        artifacts = PreprocessArtifacts(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            scaler=scaler,
            encoder=encoder,
        )
        return combined.astype(np.float32), artifacts

    def transform(self, df: pd.DataFrame, artifacts: PreprocessArtifacts) -> np.ndarray:
        numeric_columns = artifacts.numeric_columns
        categorical_columns = artifacts.categorical_columns

        numeric_data = df[numeric_columns].to_numpy(dtype=np.float32) if numeric_columns else np.empty((len(df), 0))
        if numeric_columns:
            numeric_data = artifacts.scaler.transform(np.nan_to_num(numeric_data, nan=0.0))

        cat_data = df[categorical_columns].astype(str).fillna("__MISSING__") if categorical_columns else pd.DataFrame()
        cat_encoded = artifacts.encoder.transform(cat_data) if categorical_columns else np.empty((len(df), 0))

        return np.concatenate([numeric_data, cat_encoded], axis=1).astype(np.float32)

    def inverse_transform(self, array: np.ndarray, artifacts: PreprocessArtifacts) -> pd.DataFrame:
        num_cols = artifacts.numeric_columns
        cat_cols = artifacts.categorical_columns

        num_dim = len(num_cols)
        num_part = array[:, :num_dim] if num_dim > 0 else np.empty((array.shape[0], 0))
        if num_dim > 0:
            num_part = artifacts.scaler.inverse_transform(num_part)

        cat_part = array[:, num_dim:] if cat_cols else np.empty((array.shape[0], 0))
        if cat_cols:
            decoded = artifacts.encoder.inverse_transform(cat_part)
            cat_df = pd.DataFrame(decoded, columns=cat_cols)
        else:
            cat_df = pd.DataFrame()

        num_df = pd.DataFrame(num_part, columns=num_cols) if num_dim > 0 else pd.DataFrame()
        combined = pd.concat([num_df, cat_df], axis=1)
        return combined
