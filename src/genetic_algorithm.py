import random
from common_types import DIMENSION, Case, Objective, Vector
from random_search import random_vector
from server_handler import safe_evaluate
from misc import clip


def tournament(population: list[tuple[Vector, float]], rng: random.Random, size: int = 3) -> Vector:
    selected = rng.sample(population, min(size, len(population)))
    return min(selected, key=lambda item: item[1])[0]

def make_child(parent_a: Vector, parent_b: Vector, rng: random.Random, case: Case) -> Vector:
    child: Vector = []
    if case.crossover == "blx":
        alpha = 0.3
        for left, right in zip(parent_a, parent_b):
            low = min(left, right)
            high = max(left, right)
            span = high - low
            value = rng.uniform(low - alpha * span, high + alpha * span)
            child.append(clip(value))
    else:
        weight = rng.random()
        for left, right in zip(parent_a, parent_b):
            child.append(clip(weight * left + (1.0 - weight) * right))

    for i in range(DIMENSION):
        if rng.random() < 0.35:
            child[i] = clip(child[i] + rng.gauss(0.0, case.mutation_sigma))
    return child

def genetic_algorithm(objective: Objective, case: Case) -> tuple[Vector, float, list[tuple[int, float]]]:
    rng = random.Random(case.seed)
    population_size = max(4, case.population)
    population: list[tuple[Vector, float]] = []
    history: list[tuple[int, float]] = []
    evaluations = 0

    while evaluations < min(population_size, case.budget):
        candidate = random_vector(rng)
        value = safe_evaluate(objective, candidate)
        population.append((candidate, value))
        evaluations += 1
        history.append((evaluations, min(value for _, value in population)))
        print(f"  init {evaluations:4d}/{case.budget}: current={value:.8g}, best={history[-1][1]:.8g}")

    while evaluations < case.budget:
        population.sort(key=lambda item: item[1])
        next_population = population[:2]  # a tiny bit of elitism

        while len(next_population) < population_size and evaluations < case.budget:
            parent_a = tournament(population, rng)
            parent_b = tournament(population, rng)
            child = make_child(parent_a, parent_b, rng, case)
            value = safe_evaluate(objective, child)
            next_population.append((child, value))
            evaluations += 1
            best_value = min(value for _, value in next_population + population[:2])
            history.append((evaluations, best_value))
            print(f"  eval {evaluations:4d}/{case.budget}: current={value:.8g}, best={best_value:.8g}")

        population = next_population

    best_candidate, best_value = min(population, key=lambda item: item[1])
    return best_candidate, best_value, history