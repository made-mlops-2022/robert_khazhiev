from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    numerical_features: List[str]
    features_to_drop: List[str]
    target_col: Optional[str]
    feature_to_normilize: str
