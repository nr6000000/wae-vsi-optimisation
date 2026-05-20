import math
import random
from common_types import DIMENSION, LOWER_BOUND, UPPER_BOUND, Case, Objective, Vector
from server_handler import safe_evaluate

def random_vector(rng: random.Random) -> Vector:
    return [rng.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(DIMENSION)]

def random_search(objective: Objective, case: Case) -> tuple[Vector, float, list[tuple[int, float]]]:
    rng = random.Random(case.seed)
    best_candidate: Vector | None = None
    best_value = math.inf
    history: list[tuple[int, float]] = []

    for evaluation in range(1, case.budget + 1):
        candidate = random_vector(rng)
        value = safe_evaluate(objective, candidate)
        if value < best_value:
            best_candidate = candidate
            best_value = value
        history.append((evaluation, best_value))
        print(f"  eval {evaluation:4d}/{case.budget}: current={value:.8g}, best={best_value:.8g}")

    return best_candidate or [0.0] * DIMENSION, best_value, history
