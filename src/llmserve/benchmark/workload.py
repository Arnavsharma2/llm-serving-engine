from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LengthDistribution:
    """Clipped log-normal distribution approximating right-skewed chat traffic."""

    median: int
    sigma: float
    minimum: int
    maximum: int

    def sample(self, rng: random.Random) -> int:
        value = round(rng.lognormvariate(math.log(self.median), self.sigma))
        return max(self.minimum, min(self.maximum, value))


@dataclass(frozen=True)
class LoadProfile:
    requests: int = 100
    arrival_rate: float = 2.0
    prompt: LengthDistribution = LengthDistribution(128, 0.9, 8, 1_024)
    output: LengthDistribution = LengthDistribution(64, 0.7, 4, 256)
    seed: int = 7
    vocabulary_size: int = 32_000

    @classmethod
    def from_yaml(cls, path: str | Path) -> LoadProfile:
        with Path(path).open() as handle:
            values = yaml.safe_load(handle)
        return cls(
            requests=values["requests"],
            arrival_rate=values["arrival_rate"],
            prompt=LengthDistribution(**values["prompt"]),
            output=LengthDistribution(**values["output"]),
            seed=values.get("seed", 7),
            vocabulary_size=values.get("vocabulary_size", 32_000),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_offset_s: float
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)


def generate_workload(profile: LoadProfile) -> list[RequestSpec]:
    """Generate a deterministic open-loop workload from a single seed."""

    if profile.arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")
    rng = random.Random(profile.seed)
    elapsed = 0.0
    requests: list[RequestSpec] = []
    for index in range(profile.requests):
        if index:
            elapsed += rng.expovariate(profile.arrival_rate)
        prompt_length = profile.prompt.sample(rng)
        output_length = profile.output.sample(rng)
        # Exclude common reserved ids 0-3 so the synthetic prompt does not stop early.
        tokens = tuple(rng.randrange(4, profile.vocabulary_size) for _ in range(prompt_length))
        requests.append(RequestSpec(f"request-{index:06d}", elapsed, tokens, output_length))
    return requests
