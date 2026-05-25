import random
from common_types import DIMENSION, Case, Objective, Vector
from misc import clip, random_vector
from server_handler import safe_evaluate


def differential_evolution(objective: Objective, case: Case) -> tuple[Vector, float, list[tuple[int, float]]]:
    rng = random.Random(case.seed)
    population_size = max(4, case.population)
    population: list[Vector] = []
    values: list[float] = []
    history: list[tuple[int, float]] = []
    evaluations = 0

    while evaluations < min(population_size, case.budget):
        candidate = random_vector(rng)
        value = safe_evaluate(objective, candidate)
        population.append(candidate)
        values.append(value)
        evaluations += 1
        history.append((evaluations, min(values)))
        print(f"  init {evaluations:4d}/{case.budget}: current={value:.8g}, best={history[-1][1]:.8g}")

    while evaluations < case.budget:
        for target_index in range(population_size):
            if evaluations >= case.budget:
                break

            choices = [i for i in range(population_size) if i != target_index]
            a_i, b_i, c_i = rng.sample(choices, 3)
            a = population[a_i]
            b = population[b_i]
            c = population[c_i]

            forced_dimension = rng.randrange(DIMENSION)
            trial: Vector = []
            for dimension in range(DIMENSION):
                if rng.random() < case.crossover_rate or dimension == forced_dimension:
                    value = a[dimension] + case.differential_weight * (b[dimension] - c[dimension])
                    trial.append(clip(value))
                else:
                    trial.append(population[target_index][dimension])

            trial_value = safe_evaluate(objective, trial)
            evaluations += 1
            if trial_value <= values[target_index]:
                population[target_index] = trial
                values[target_index] = trial_value

            history.append((evaluations, min(values)))
            print(f"  eval {evaluations:4d}/{case.budget}: current={trial_value:.8g}, best={history[-1][1]:.8g}")

    best_index = min(range(population_size), key=lambda i: values[i])
    return population[best_index], values[best_index], history
