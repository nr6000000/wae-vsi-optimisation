import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import optuna

from common_types import Case
from differential_evolution import differential_evolution
from genetic_algorithm import genetic_algorithm
from server_handler import make_server_objective


def median(values: list[float]) -> float:
    return statistics.median(values)


def iqr(values: list[float]) -> float:
    if len(values) < 4:
        return math.nan
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def run_method_once(method: str, objective, case: Case) -> float:
    if method == "ga":
        _, best_value, _ = genetic_algorithm(objective, min, case)
        return float(best_value)

    if method in {"de", "de_rand_1_bin"}:
        _, best_value, _ = differential_evolution(objective, min, case)
        return float(best_value)

    raise ValueError(f"Optuna tuning supports only GA and DE, got: {method}")


def case_from_trial(method: str, trial: optuna.Trial, seed: int, budget: int) -> Case:
    if method == "ga":
        population = trial.suggest_categorical("population", [10, 20, 40])
        mutation_sigma = trial.suggest_float("mutation_sigma", 0.15, 1.0)
        crossover = trial.suggest_categorical("crossover", ["arithmetic", "blx"])

        return Case(
            name=f"optuna_ga_t{trial.number}_s{seed}",
            method="ga",
            budget=budget,
            seed=seed,
            population=int(population),
            mutation_sigma=float(mutation_sigma),
            crossover=str(crossover),
        )

    if method == "de":
        population = trial.suggest_categorical("population", [10, 20, 40])
        differential_weight = trial.suggest_float("differential_weight", 0.3, 1.0)
        crossover_rate = trial.suggest_float("crossover_rate", 0.4, 1.0)

        return Case(
            name=f"optuna_de_t{trial.number}_s{seed}",
            method="de",
            budget=budget,
            seed=seed,
            population=int(population),
            differential_weight=float(differential_weight),
            crossover_rate=float(crossover_rate),
        )

    raise ValueError(f"Unknown method: {method}")


def tune_method(
    method: str,
    objective,
    seeds: list[int],
    budget: int,
    n_trials: int,
    study_seed: int,
    out_dir: Path,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=study_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def tuning_objective(trial: optuna.Trial) -> float:
        values: list[float] = []

        print("\n" + "-" * 72)
        print(f"Optuna {method.upper()} trial {trial.number}")

        for seed in seeds:
            case = case_from_trial(method, trial, seed, budget)
            value = run_method_once(method, objective, case)
            values.append(value)
            print(f"  seed={seed}: best J(x)={value:.12g}")

        score = median(values)
        trial.set_user_attr("values_by_seed", values)
        trial.set_user_attr("iqr", iqr(values))
        print(f"  trial median={score:.12g}, iqr={trial.user_attrs['iqr']}")
        return score

    study.optimize(tuning_objective, n_trials=n_trials)

    method_dir = out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    write_trials_csv(study, method_dir / "trials.csv")
    write_best_config(study, method, budget, seeds, method_dir / "best_config.md")
    make_plots(study, method, method_dir / "plots")

    return study


def write_trials_csv(study: optuna.Study, path: Path) -> None:
    rows: list[dict[str, object]] = []

    for trial in study.trials:
        row: dict[str, object] = {
            "trial": trial.number,
            "state": str(trial.state),
            "value_median": trial.value,
            "iqr": trial.user_attrs.get("iqr", ""),
            "values_by_seed": trial.user_attrs.get("values_by_seed", ""),
        }
        for key, value in trial.params.items():
            row[key] = value
        rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_config(study: optuna.Study, method: str, budget: int, seeds: list[int], path: Path) -> None:
    best = study.best_trial

    lines = [
        f"# Best Optuna configuration for {method.upper()}",
        "",
        "Optuna was used only for tuning hyperparameters.",
        "The final comparison should be run later with fixed parameters and separate final cases.",
        "",
        f"- method: `{method}`",
        f"- tuning budget per run: `{budget}`",
        f"- tuning seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- best median J(x): `{best.value}`",
        f"- IQR for best trial: `{best.user_attrs.get('iqr', '')}`",
        "",
        "## Parameters",
        "",
    ]

    for key, value in best.params.items():
        lines.append(f"- `{key}` = `{value}`")

    lines += ["", "## CSV template", ""]

    lines.append("case,method,budget,seed,population,mutation_sigma,crossover,differential_weight,crossover_rate")
    if method == "ga":
        lines.append(
            "ga_selected,ga,FINAL_BUDGET,FINAL_SEED,"
            f"{best.params['population']},{best.params['mutation_sigma']},{best.params['crossover']},,"
        )
    else:
        lines.append(
            "de_selected,de,FINAL_BUDGET,FINAL_SEED,"
            f"{best.params['population']},,,{best.params['differential_weight']},{best.params['crossover_rate']}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(study: optuna.Study, method: str, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        return

    trial_numbers = [trial.number for trial in completed]
    values = [float(trial.value) for trial in completed]

    best_so_far: list[float] = []
    current_best = math.inf
    for value in values:
        current_best = min(current_best, value)
        best_so_far.append(current_best)

    plt.figure(figsize=(8, 5))
    plt.plot(trial_numbers, values, marker="o", label="trial median")
    plt.plot(trial_numbers, best_so_far, marker="o", label="best so far")
    plt.xlabel("Optuna trial")
    plt.ylabel("median best J(x) over tuning seeds")
    plt.title(f"Optuna tuning: {method}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "optuna_trials.png", dpi=160)
    plt.close()

    for param_name in sorted(study.best_params):
        xs = [trial.params.get(param_name) for trial in completed]
        if any(x is None for x in xs):
            continue

        plt.figure(figsize=(7, 5))
        plt.scatter(xs, values)
        plt.xlabel(param_name)
        plt.ylabel("median best J(x)")
        plt.title(f"Optuna tuning: {method}, {param_name}")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        safe_name = param_name.replace("/", "_")
        plt.savefig(plot_dir / f"param_{safe_name}.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune GA/DE hyperparameters with Optuna.")
    parser.add_argument("--method", choices=["ga", "de", "both"], default="both")
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seeds", default="316018,348556,1001")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--study-seed", type=int, default=20260603)
    parser.add_argument("--out", type=Path, default=Path("results/optuna_tuning"))

    args = parser.parse_args()

    objective = make_server_objective(args.address, args.port)
    seeds = parse_seeds(args.seeds)
    methods = ["ga", "de"] if args.method == "both" else [args.method]

    for method in methods:
        print("\n" + "=" * 72)
        print(f"Optuna tuning for {method.upper()}")
        print(f"budget={args.budget}, seeds={seeds}, n_trials={args.n_trials}")
        study = tune_method(
            method=method,
            objective=objective,
            seeds=seeds,
            budget=args.budget,
            n_trials=args.n_trials,
            study_seed=args.study_seed,
            out_dir=args.out,
        )
        print(f"Best value for {method}: {study.best_value}")
        print(f"Best params for {method}: {study.best_params}")


if __name__ == "__main__":
    main()
