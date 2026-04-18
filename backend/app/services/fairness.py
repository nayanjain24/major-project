from __future__ import annotations

from typing import Dict

import pandas as pd


class FairnessService:
    def group_balance(self, df: pd.DataFrame, sensitive_column: str) -> Dict[str, float]:
        counts = df[sensitive_column].value_counts(normalize=True, dropna=False)
        return counts.round(4).to_dict()
