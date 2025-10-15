from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    DATA_DIR: Path = Path("./data")
    DATASET_PATH: Path = DATA_DIR / "movie_metadata_with_reviews.csv"


@dataclass(frozen=True)
class SentimentConfig:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 256
    batch_size: int = 64
    epochs: int = 2
    lr: float = 1e-5


@dataclass(frozen=True)
class HybridWeights:
    w_content: float = 0.4
    w_cf: float = 0.4
    w_sentiment: float = 0.2


@dataclass(frozen=True)
class EvalConfig:
    ks: tuple[int, ...] = (5, 10)


@dataclass(frozen=True)
class SystemConfig:
    seed: int = 19
    device: str = "cpu"  # "cuda" - 5h or "cpu" 8+h?
