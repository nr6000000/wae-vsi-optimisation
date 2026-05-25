import random
from common_types import DIMENSION, LOWER_BOUND, UPPER_BOUND, Vector


def clip(value: float) -> float:
    return min(max(value, LOWER_BOUND), UPPER_BOUND)

def random_vector(rng: random.Random) -> Vector:
    return [rng.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(DIMENSION)]