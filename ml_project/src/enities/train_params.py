from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="GaussianNB")
    random_state: int = field(default=255)
    n_estimators: int = field(default=200)
    max_depth: int = field(default=1)
    lr: float = field(default=1e-1)
