from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class MetricsService:
    def distribution_similarity(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for col in original.columns:
            if pd.api.types.is_numeric_dtype(original[col]):
                try:
                    stat, _ = ks_2samp(original[col].dropna(), synthetic[col].dropna())
                    scores[f"ks_{col}"] = float(1.0 - stat)
                except ValueError:
                    scores[f"ks_{col}"] = 0.0
        if scores:
            scores["distribution_similarity"] = float(np.mean(list(scores.values())))
        return scores
