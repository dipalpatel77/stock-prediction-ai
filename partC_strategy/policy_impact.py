from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class PolicyEvent:
    id: str
    date: pd.Timestamp
    headline: str
    policy_type: str
    expected_direction: str
    severity: float = 1.0
    confidence: float = 0.5
    effective_date: Optional[pd.Timestamp] = None
    notes: Optional[str] = None

class PolicyImpactModel:
    """Minimal stub: estimate no change and return df unchanged."""
    def __init__(self, clamp_min: float = 0.05, clamp_max: float = 3.0):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def apply_policy_adjustments(self, df: pd.DataFrame, events: List[PolicyEvent], **kwargs) -> pd.DataFrame:
        df = df.copy()
        if 'Policy_Impact_Score' not in df.columns:
            df['Policy_Impact_Score'] = 0.0
        if 'Policy_Adjustment' not in df.columns:
            df['Policy_Adjustment'] = 1.0
        return df