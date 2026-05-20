from dataclasses import dataclass
from typing import Callable


Vector = list[float]
Objective = Callable[[Vector], float]

@dataclass
class Case:
    name: str
    method: str
    budget: int
    seed: int

BAD_VALUE = 1_000_000.0
LOWER_BOUND = -4.0
UPPER_BOUND = 4.0
DIMENSION = 4
