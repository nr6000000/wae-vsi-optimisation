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
    population: int = 10
    mutation_sigma: float = 0.7
    crossover: str = "arithmetic"
    differential_weight: float = 0.7
    crossover_rate: float = 0.9

BAD_VALUE = 1_000_000.0
LOWER_BOUND = -4.0
UPPER_BOUND = 4.0
DIMENSION = 4
