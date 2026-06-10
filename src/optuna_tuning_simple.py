import argparse
import csv
import statistics
from pathlib import Path

import optuna

from common_types import Case
from differential_evolution import differential_evolution
from genetic_algorithm import genetic_algorithm
from server_handler import make_server_objective


def parse_seeds(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_ga(objective, seed, budget, population, mutation_sigma, crossover):
    case = Case(
        name=f"ga_tuning_s{seed}",
        method="ga",
        budget=budget,
        seed=seed,
        population=population,
        mutation_sigma=mutation_sigma,
        crossover=crossover,
    )
    _best_x, best_value, _history = genetic_algorithm(objective, min, case)
    return float(best_value)


def run_de(objective, seed, budget, population, differential_weight, crossover_rate):
    case = Case(
        name=f"de_tuning_s{seed}",
        method="de",
        budget=budget,
        seed=seed,
        population=population,
        differential_weight=differential_weight,
        crossover_rate=crossover_rate,
    )
    _best_x, best_value, _history = differential_evolution(objective, min, case)
    return float(best_value)


def tune_ga(objective, seeds, budget, n_trials):
    def objective_for_optuna(trial):
        population = trial.suggest_categorical("population", [10, 20, 40])
        mutation_sigma = trial.suggest_float("mutation_sigma", 0.15, 1.0)
        crossover = trial.suggest_categorical("crossover", ["arithmetic", "blx"])

        values = []
        print()
        print(f"GA trial {trial.number}: population={population}, sigma={mutation_sigma:.4f}, crossover={crossover}")

        for seed in seeds:
            value = run_ga(objective, seed, budget, population, mutation_sigma, crossover)
            values.append(value)
            print(f"  seed={seed}, best J={value:.12g}")

        score = statistics.median(values)
        trial.set_user_attr("values", values)
        print(f"  median={score:.12g}")
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_for_optuna, n_trials=n_trials)
    return study


def tune_de(objective, seeds, budget, n_trials):
    def objective_for_optuna(trial):
        population = trial.suggest_categorical("population", [10, 20, 40])
        differential_weight = trial.suggest_float("differential_weight", 0.3, 1.0)
        crossover_rate = trial.suggest_float("crossover_rate", 0.4, 1.0)

        values = []
        print()
        print(
            f"DE trial {trial.number}: population={population}, "
            f"F={differential_weight:.4f}, CR={crossover_rate:.4f}"
        )

        for seed in seeds:
            value = run_de(objective, seed, budget, population, differential_weight, crossover_rate)
            values.append(value)
            print(f"  seed={seed}, best J={value:.12g}")

        score = statistics.median(values)
        trial.set_user_attr("values", values)
        print(f"  median={score:.12g}")
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_for_optuna, n_trials=n_trials)
    return study


def save_trials(study, path):
    rows = []

    for trial in study.trials:
        row = {
            "trial": trial.number,
            "value_median": trial.value,
            "values_by_seed": trial.user_attrs.get("values", ""),
        }
        row.update(trial.params)
        rows.append(row)

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_best_config(method, study, path, budget, seeds):
    best = study.best_trial

    lines = []
    lines.append(f"# Best config for {method.upper()}")
    lines.append("")
    lines.append("Optuna was used only to tune hyperparameters.")
    lines.append("These parameters should be copied/frozen before the final comparison.")
    lines.append("")
    lines.append(f"- tuning budget: `{budget}`")
    lines.append(f"- tuning seeds: `{', '.join(str(seed) for seed in seeds)}`")
    lines.append(f"- best median J(x): `{best.value}`")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")

    for key, value in best.params.items():
        lines.append(f"- `{key}` = `{value}`")

    lines.append("")
    lines.append("## Rounded values for final CSV")
    lines.append("")

    if method == "ga":
        lines.append("GA final row example:")
        lines.append("")
        lines.append(
            f"`ga_final,ga,FINAL_BUDGET,FINAL_SEED,"
            f"{best.params['population']},{float(best.params['mutation_sigma']):.2f},"
            f"{best.params['crossover']},,`"
        )
    else:
        lines.append("DE final row example:")
        lines.append("")
        lines.append(
            f"`de_final,de,FINAL_BUDGET,FINAL_SEED,"
            f"{best.params['population']},,,"
            f"{float(best.params['differential_weight']):.2f},"
            f"{float(best.params['crossover_rate']):.2f}`"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_selected_summary(ga_study, de_study, path):
    ga = ga_study.best_trial.params
    de = de_study.best_trial.params

    rows = [
        {
            "method": "ga",
            "population": ga["population"],
            "mutation_sigma": round(float(ga["mutation_sigma"]), 2),
            "crossover": ga["crossover"],
            "differential_weight": "",
            "crossover_rate": "",
            "best_median_value": ga_study.best_value,
        },
        {
            "method": "de",
            "population": de["population"],
            "mutation_sigma": "",
            "crossover": "",
            "differential_weight": round(float(de["differential_weight"]), 2),
            "crossover_rate": round(float(de["crossover_rate"]), 2),
            "best_median_value": de_study.best_value,
        },
    ]

    fields = [
        "method",
        "population",
        "mutation_sigma",
        "crossover",
        "differential_weight",
        "crossover_rate",
        "best_median_value",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Simple Optuna tuning for GA and DE.")
    parser.add_argument("--method", choices=["ga", "de", "both"], default="both")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seeds", default="316018,348556,1001")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--out", type=Path, default=Path("results/optuna_tuning"))

    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    objective = make_server_objective(args.address, args.port)

    ga_study = None
    de_study = None

    if args.method in {"ga", "both"}:
        print("=" * 72)
        print("Tuning GA")
        ga_study = tune_ga(objective, seeds, args.budget, args.n_trials)
        save_trials(ga_study, args.out / "ga" / "trials.csv")
        save_best_config("ga", ga_study, args.out / "ga" / "best_config.md", args.budget, seeds)
        print(f"Best GA params: {ga_study.best_params}")
        print(f"Best GA median: {ga_study.best_value}")

    if args.method in {"de", "both"}:
        print("=" * 72)
        print("Tuning DE")
        de_study = tune_de(objective, seeds, args.budget, args.n_trials)
        save_trials(de_study, args.out / "de" / "trials.csv")
        save_best_config("de", de_study, args.out / "de" / "best_config.md", args.budget, seeds)
        print(f"Best DE params: {de_study.best_params}")
        print(f"Best DE median: {de_study.best_value}")

    if ga_study is not None and de_study is not None:
        save_selected_summary(ga_study, de_study, args.out / "selected_hyperparameters.csv")

    print()
    print("Created results in:")
    print(args.out)


if __name__ == "__main__":
    main()
