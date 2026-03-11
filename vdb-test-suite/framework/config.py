from dataclasses import dataclass

@dataclass(slots=True)
class RunConfig:
    port: int = 8080
    n_search: int = 200
    warmup: int = 10
    concurrency: int = 8
    duration_s: float = 5.0
    batch_size: int = 5000
    startup_timeout_s: float = 20.0
    settle_time_s: float = 1.0
    metric: str = "cosine"
    seed: int = 42
