from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ProfileResult:
    rows: int
    cols: int
    columns: List[dict]
    missing_by_column: Dict[str, float]
    correlation_matrix: List[List[float]]
    correlation_labels: List[str]
    categorical_cardinality: Dict[str, int]
    metadata: Dict[str, str]


class DataProfiler:
    def profile(self, df: pd.DataFrame) -> ProfileResult:
        rows, cols = df.shape
        missing_by_column = (df.isna().mean() * 100.0).round(2).to_dict()

        columns = []
        categorical_cardinality: Dict[str, int] = {}
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            unique = int(series.nunique(dropna=True))
            profile = {
                "name": col,
                "dtype": dtype,
                "missing_pct": float(missing_by_column.get(col, 0.0)),
                "unique": unique,
            }
            if pd.api.types.is_numeric_dtype(series):
                profile.update(
                    {
                        "mean": float(series.mean()) if series.notna().any() else None,
                        "std": float(series.std()) if series.notna().any() else None,
                        "min": float(series.min()) if series.notna().any() else None,
                        "max": float(series.max()) if series.notna().any() else None,
                    }
                )
            else:
                categorical_cardinality[col] = unique
            columns.append(profile)

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr().fillna(0.0)
            correlation_matrix = corr.values.round(3).tolist()
            correlation_labels = corr.columns.tolist()
        else:
            correlation_matrix = []
            correlation_labels = []

        metadata = {
            "numeric_features": str(numeric_df.shape[1]),
            "categorical_features": str(len(categorical_cardinality)),
        }

        return ProfileResult(
            rows=rows,
            cols=cols,
            columns=columns,
            missing_by_column=missing_by_column,
            correlation_matrix=correlation_matrix,
            correlation_labels=correlation_labels,
            categorical_cardinality=categorical_cardinality,
            metadata=metadata,
        )
