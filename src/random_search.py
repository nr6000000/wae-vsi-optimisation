import math
import random
from typing import Callable
from common_types import DIMENSION, Case, Objective, Vector
from server_handler import safe_evaluate
from misc import random_vector

def random_search(objective: Objective, comparator, case: Case) -> tuple[Vector, float, list[tuple[int, float]]]:
    rng = random.Random(case.seed)
    best_candidate: Vector | None = None
    best_value = math.inf if comparator is min else -math.inf
    history: list[tuple[int, float]] = []

    for evaluation in range(1, case.budget + 1):
        candidate = random_vector(rng)
        value = safe_evaluate(objective, candidate)

        if best_candidate is None:
            best_candidate = candidate
            best_value = value
        else:
            best_candidate, best_value = comparator(
                ((candidate, value), (best_candidate, best_value)),
                key=lambda item: item[1],
            )

        history.append((evaluation, best_value))
        print(f"  eval {evaluation:4d}/{case.budget}: current={value:.8g}, best={best_value:.8g}")

    return best_candidate or [0.0] * DIMENSION, best_value, history
