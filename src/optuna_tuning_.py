import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import optuna

from common_types import Case
from differential_evolution import differential_evolution
from genetic_algorithm import genetic_algorithm
from server_handler import make_server_objective


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds


def run_ga(objective, seed, budget, population, mutation_sigma, crossover):
    case = Case(
        name=f"ga_optuna_seed_{seed}",
        method="ga",
        budget=budget,
        seed=seed,
        population=population,
        mutation_sigma=mutation_sigma,
        crossover=crossover,
    )
    _, best_value, _ = genetic_algorithm(objective, min, case)
    return float(best_value)


def run_de(objective, seed, budget, population, differential_weight, crossover_rate):
    case = Case(
        name=f"de_optuna_seed_{seed}",
        method="de",
        budget=budget,
        seed=seed,
        population=population,
        differential_weight=differential_weight,
        crossover_rate=crossover_rate,
    )
    _, best_value, _ = differential_evolution(objective, min, case)
    return float(best_value)


def tune_ga(objective, seeds, budget, n_trials, out_dir):
    def objective_for_optuna(trial):
        population = trial.suggest_categorical("population", [10, 20, 40])
        mutation_sigma = trial.suggest_float("mutation_sigma", 0.15, 1.0)
        crossover = trial.suggest_categorical("crossover", ["arithmetic", "blx"])

        values = []
        print("\nGA trial", trial.number, population, mutation_sigma, crossover)

        for seed in seeds:
            value = run_ga(objective, seed, budget, population, mutation_sigma, crossover)
            values.append(value)
            print("  seed", seed, "best", value)

        result = statistics.median(values)
        trial.set_user_attr("values_by_seed", values)
        print("  median", result)
        return result

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_for_optuna, n_trials=n_trials)

    save_study("ga", study, budget, seeds, out_dir / "ga")
    return study


def tune_de(objective, seeds, budget, n_trials, out_dir):
    def objective_for_optuna(trial):
        population = trial.suggest_categorical("population", [10, 20, 40])
        differential_weight = trial.suggest_float("differential_weight", 0.3, 1.0)
        crossover_rate = trial.suggest_float("crossover_rate", 0.4, 1.0)

        values = []
        print("\nDE trial", trial.number, population, differential_weight, crossover_rate)

        for seed in seeds:
            value = run_de(objective, seed, budget, population, differential_weight, crossover_rate)
            values.append(value)
            print("  seed", seed, "best", value)

        result = statistics.median(values)
        trial.set_user_attr("values_by_seed", values)
        print("  median", result)
        return result

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_for_optuna, n_trials=n_trials)

    save_study("de", study, budget, seeds, out_dir / "de")
    return study


def save_study(method, study, budget, seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "value_median": trial.value,
            "values_by_seed": trial.user_attrs.get("values_by_seed", ""),
        }
        row.update(trial.params)
        rows.append(row)

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with (out_dir / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    best = study.best_trial
    lines = [
        f"# Best config for {method.upper()}",
        "",
        f"Budget used for tuning: `{budget}`",
        f"Seeds used for tuning: `{', '.join(str(seed) for seed in seeds)}`",
        f"Best median J(x): `{best.value}`",
        "",
        "## Parameters",
        "",
    ]

    for key, value in best.params.items():
        lines.append(f"- `{key}` = `{value}`")

    lines.append("")
    lines.append("These parameters should be copied into final comparison cases.")
    (out_dir / "best_config.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    trial_numbers = [trial.number for trial in study.trials if trial.value is not None]
    values = [float(trial.value) for trial in study.trials if trial.value is not None]

    best_so_far = []
    current = None
    for value in values:
        current = value if current is None else min(current, value)
        best_so_far.append(current)

    plt.figure(figsize=(8, 5))
    plt.plot(trial_numbers, values, marker="o", label="trial")
    plt.plot(trial_numbers, best_so_far, marker="o", label="best so far")
    plt.xlabel("trial")
    plt.ylabel("median best J(x)")
    plt.title(f"Optuna tuning - {method}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "optuna_trials.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ga", "de", "both"], default="both")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seeds", default="316018,348556,1001")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--out", type=Path, default=Path("results/optuna_tuning"))
    args = parser.parse_args()

    objective = make_server_objective(args.address, args.port)
    seeds = parse_seeds(args.seeds)

    if args.method in ("ga", "both"):
        tune_ga(objective, seeds, args.budget, args.n_trials, args.out)

    if args.method in ("de", "both"):
        tune_de(objective, seeds, args.budget, args.n_trials, args.out)


if __name__ == "__main__":
    main()
