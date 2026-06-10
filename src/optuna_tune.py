import argparse
import csv
import math
import shutil
import statistics
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState

from common_types import Case
from differential_evolution import differential_evolution
from genetic_algorithm import genetic_algorithm
from server_handler import make_server_objective


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.17g}"


def successful_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None and math.isfinite(float(t.value))
    ]


def clean_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------

def run_ga(objective, seed: int, budget: int, population: int, mutation_sigma: float, crossover: str) -> float:
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


def run_de(
    objective,
    seed: int,
    budget: int,
    population: int,
    differential_weight: float,
    crossover_rate: float,
) -> float:
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


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------

def enqueue_ga_baselines(study: optuna.Study) -> None:
    """
    These baseline trials make the tuning less stupid for tiny n_trials.

    With --n-trials 4, GA will test:
      - arithmetic and blx
      - population 10 and 20
      - mutation_sigma 0.40

    So yes: arithmetic will be saved in trials.csv when n_trials >= 1,
    and both arithmetic + blx will be saved when n_trials >= 2.
    """
    baselines = [
        {"population": 10, "mutation_sigma": 0.40, "crossover": "arithmetic"},
        {"population": 10, "mutation_sigma": 0.40, "crossover": "blx"},
        {"population": 20, "mutation_sigma": 0.40, "crossover": "arithmetic"},
        {"population": 20, "mutation_sigma": 0.40, "crossover": "blx"},
    ]
    for params in baselines:
        study.enqueue_trial(params)


def enqueue_de_baselines(study: optuna.Study) -> None:
    baselines = [
        {"population": 10, "differential_weight": 0.50, "crossover_rate": 0.90},
        {"population": 20, "differential_weight": 0.50, "crossover_rate": 0.90},
        {"population": 10, "differential_weight": 0.70, "crossover_rate": 0.90},
        {"population": 20, "differential_weight": 0.70, "crossover_rate": 0.90},
    ]
    for params in baselines:
        study.enqueue_trial(params)


def tune_ga(
    objective,
    seeds: list[int],
    budget: int,
    n_trials: int,
    optuna_seed: int,
    enqueue_baselines: bool,
) -> optuna.Study:
    def objective_for_optuna(trial: optuna.Trial) -> float:
        population = trial.suggest_categorical("population", [10, 20, 40])
        mutation_sigma = trial.suggest_float("mutation_sigma", 0.15, 1.0)
        crossover = trial.suggest_categorical("crossover", ["arithmetic", "blx"])

        values = []
        seed_values = OrderedDict()

        print()
        print(
            f"GA trial {trial.number}: "
            f"population={population}, sigma={mutation_sigma:.6f}, crossover={crossover}"
        )

        for seed in seeds:
            value = run_ga(objective, seed, budget, population, mutation_sigma, crossover)
            values.append(value)
            seed_values[str(seed)] = value
            print(f"  seed={seed}, best J={value:.12g}")

        score = statistics.median(values)
        trial.set_user_attr("values", values)
        trial.set_user_attr("seed_values", dict(seed_values))
        print(f"  median={score:.12g}")
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    if enqueue_baselines:
        enqueue_ga_baselines(study)

    study.optimize(objective_for_optuna, n_trials=n_trials)
    return study


def tune_de(
    objective,
    seeds: list[int],
    budget: int,
    n_trials: int,
    optuna_seed: int,
    enqueue_baselines: bool,
) -> optuna.Study:
    def objective_for_optuna(trial: optuna.Trial) -> float:
        population = trial.suggest_categorical("population", [10, 20, 40])
        differential_weight = trial.suggest_float("differential_weight", 0.3, 1.0)
        crossover_rate = trial.suggest_float("crossover_rate", 0.4, 1.0)

        values = []
        seed_values = OrderedDict()

        print()
        print(
            f"DE trial {trial.number}: "
            f"population={population}, F={differential_weight:.6f}, CR={crossover_rate:.6f}"
        )

        for seed in seeds:
            value = run_de(objective, seed, budget, population, differential_weight, crossover_rate)
            values.append(value)
            seed_values[str(seed)] = value
            print(f"  seed={seed}, best J={value:.12g}")

        score = statistics.median(values)
        trial.set_user_attr("values", values)
        trial.set_user_attr("seed_values", dict(seed_values))
        print(f"  median={score:.12g}")
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    if enqueue_baselines:
        enqueue_de_baselines(study)

    study.optimize(objective_for_optuna, n_trials=n_trials)
    return study


# ---------------------------------------------------------------------------
# CSV / Markdown saving
# ---------------------------------------------------------------------------

def save_trials(study: optuna.Study, path: Path) -> None:
    rows = []

    for trial in study.trials:
        seed_values = trial.user_attrs.get("seed_values", {})
        seed_values_text = ";".join(
            f"{seed}:{fmt_float(value)}" for seed, value in seed_values.items()
        )

        row = OrderedDict()
        row["trial"] = trial.number
        row["state"] = trial.state.name
        row["value_median"] = fmt_float(trial.value)
        row["values_by_seed"] = seed_values_text

        for key, value in trial.params.items():
            row[key] = value

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


def save_param_group_summary(study: optuna.Study, param: str, path: Path) -> None:
    groups: OrderedDict[str, list[float]] = OrderedDict()

    for trial in successful_trials(study):
        if param not in trial.params:
            continue
        key = str(trial.params[param])
        groups.setdefault(key, []).append(float(trial.value))

    rows = []
    for key, values in groups.items():
        rows.append(
            {
                param: key,
                "count": len(values),
                "best_median_value": fmt_float(min(values)),
                "median_of_trial_medians": fmt_float(statistics.median(values)),
                "mean_of_trial_medians": fmt_float(statistics.mean(values)),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = [param, "count", "best_median_value", "median_of_trial_medians", "mean_of_trial_medians"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_best_config(method: str, study: optuna.Study, path: Path, budget: int, seeds: list[int], n_trials: int) -> None:
    best = study.best_trial

    lines = []
    lines.append(f"# Best config for {method.upper()}")
    lines.append("")
    lines.append("Optuna was used only to tune hyperparameters.")
    lines.append("These parameters should be copied/frozen before the final comparison.")
    lines.append("")
    lines.append(f"- tuning budget: `{budget}`")
    lines.append(f"- tuning seeds: `{', '.join(str(seed) for seed in seeds)}`")
    lines.append(f"- Optuna trials: `{n_trials}`")
    lines.append(f"- best trial number: `{best.number}`")
    lines.append(f"- best median J(x): `{best.value}`")
    lines.append("")

    lines.append("## Search space")
    lines.append("")
    if method == "ga":
        lines.append("- `population`: `[10, 20, 40]`")
        lines.append("- `mutation_sigma`: `[0.15, 1.0]`")
        lines.append("- `crossover`: `['arithmetic', 'blx']`")
    else:
        lines.append("- `population`: `[10, 20, 40]`")
        lines.append("- `differential_weight`: `[0.3, 1.0]`")
        lines.append("- `crossover_rate`: `[0.4, 1.0]`")
    lines.append("")

    lines.append("## Parameters")
    lines.append("")
    for key, value in best.params.items():
        lines.append(f"- `{key}` = `{value}`")
    lines.append("")

    lines.append("## Top trials")
    lines.append("")
    complete = sorted(successful_trials(study), key=lambda t: float(t.value))
    for trial in complete[: min(5, len(complete))]:
        params = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        lines.append(f"- trial `{trial.number}`: median `{trial.value}`; {params}")
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


def save_selected_summary(ga_study: optuna.Study, de_study: optuna.Study, path: Path) -> None:
    ga = ga_study.best_trial.params
    de = de_study.best_trial.params

    rows = [
        {
            "method": "ga",
            "population": ga["population"],
            "mutation_sigma": fmt_float(float(ga["mutation_sigma"])),
            "mutation_sigma_rounded": f"{float(ga['mutation_sigma']):.2f}",
            "crossover": ga["crossover"],
            "differential_weight": "",
            "differential_weight_rounded": "",
            "crossover_rate": "",
            "crossover_rate_rounded": "",
            "best_median_value": fmt_float(ga_study.best_value),
        },
        {
            "method": "de",
            "population": de["population"],
            "mutation_sigma": "",
            "mutation_sigma_rounded": "",
            "crossover": "",
            "differential_weight": fmt_float(float(de["differential_weight"])),
            "differential_weight_rounded": f"{float(de['differential_weight']):.2f}",
            "crossover_rate": fmt_float(float(de["crossover_rate"])),
            "crossover_rate_rounded": f"{float(de['crossover_rate']):.2f}",
            "best_median_value": fmt_float(de_study.best_value),
        },
    ]

    fields = [
        "method",
        "population",
        "mutation_sigma",
        "mutation_sigma_rounded",
        "crossover",
        "differential_weight",
        "differential_weight_rounded",
        "crossover_rate",
        "crossover_rate_rounded",
        "best_median_value",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_run_metadata(path: Path, args: argparse.Namespace, seeds: list[int]) -> None:
    lines = [
        "# Optuna tuning run metadata",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- method: `{args.method}`",
        f"- budget: `{args.budget}`",
        f"- seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- n_trials per method: `{args.n_trials}`",
        f"- optuna_seed: `{args.optuna_seed}`",
        f"- output: `{args.out}`",
        f"- enqueue_baselines: `{not args.no_enqueue_baselines}`",
        "",
        "Note: `--n-trials` is counted per method. With `--method both --n-trials 12`, this runs 12 GA trials and 12 DE trials.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_optimization_history_plot(study: optuna.Study, path: Path, title: str) -> None:
    trials = successful_trials(study)
    if not trials:
        return

    numbers = [trial.number for trial in trials]
    values = [float(trial.value) for trial in trials]

    best_so_far = []
    current_best = float("inf")
    for value in values:
        current_best = min(current_best, value)
        best_so_far.append(current_best)

    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(numbers, values, marker="o", label="trial median J(x)")
    plt.plot(numbers, best_so_far, marker="o", label="best so far")
    plt.xlabel("Trial")
    plt.ylabel("Median J(x)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_categorical_param_plot(study: optuna.Study, param: str, path: Path, title: str) -> None:
    groups: OrderedDict[str, list[float]] = OrderedDict()

    for trial in successful_trials(study):
        if param not in trial.params:
            continue
        key = str(trial.params[param])
        groups.setdefault(key, []).append(float(trial.value))

    if not groups:
        return

    labels = list(groups.keys())
    data = [groups[label] for label in labels]

    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showmeans=True)

    for i, values in enumerate(data, start=1):
        plt.scatter([i] * len(values), values, alpha=0.75)

    plt.xlabel(param)
    plt.ylabel("Median J(x)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_numeric_param_plot(study: optuna.Study, param: str, path: Path, title: str) -> None:
    xs = []
    ys = []

    for trial in successful_trials(study):
        if param not in trial.params:
            continue
        xs.append(float(trial.params[param]))
        ys.append(float(trial.value))

    if not xs:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, alpha=0.8)

    best_trial = study.best_trial
    if param in best_trial.params:
        plt.scatter([float(best_trial.params[param])], [float(best_trial.value)], marker="*", s=180, label="best trial")
        plt.legend()

    plt.xlabel(param)
    plt.ylabel("Median J(x)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_plots(method: str, study: optuna.Study, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_optimization_history_plot(
        study,
        plots_dir / "optimization_history.png",
        f"{method.upper()} Optuna optimization history",
    )

    if method == "ga":
        save_categorical_param_plot(
            study,
            "population",
            plots_dir / "param_population.png",
            "GA median J(x) by population",
        )
        save_numeric_param_plot(
            study,
            "mutation_sigma",
            plots_dir / "param_mutation_sigma.png",
            "GA median J(x) by mutation_sigma",
        )
        save_categorical_param_plot(
            study,
            "crossover",
            plots_dir / "param_crossover.png",
            "GA median J(x) by crossover",
        )
    else:
        save_categorical_param_plot(
            study,
            "population",
            plots_dir / "param_population.png",
            "DE median J(x) by population",
        )
        save_numeric_param_plot(
            study,
            "differential_weight",
            plots_dir / "param_differential_weight.png",
            "DE median J(x) by differential_weight",
        )
        save_numeric_param_plot(
            study,
            "crossover_rate",
            plots_dir / "param_crossover_rate.png",
            "DE median J(x) by crossover_rate",
        )


def save_summaries(method: str, study: optuna.Study, out_dir: Path) -> None:
    summaries_dir = out_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    save_param_group_summary(study, "population", summaries_dir / "population_summary.csv")

    if method == "ga":
        save_param_group_summary(study, "crossover", summaries_dir / "crossover_summary.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Optuna tuning for GA and DE, with plots.")
    parser.add_argument("--method", choices=["ga", "de", "both"], default="both")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seeds", default="316018,348556,1001")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--optuna-seed", type=int, default=2026)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--out", type=Path, default=Path("results/optuna_tuning"))
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove the output directory before writing. By default, the output directory is deleted first.",
    )
    parser.add_argument(
        "--no-enqueue-baselines",
        action="store_true",
        help="Do not enqueue fixed baseline trials. By default, baseline trials force arithmetic/blx coverage for small GA runs.",
    )

    args = parser.parse_args()

    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1")

    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")

    if not args.no_clean:
        clean_output_dir(args.out)
    else:
        args.out.mkdir(parents=True, exist_ok=True)

    save_run_metadata(args.out / "run_metadata.md", args, seeds)

    objective = make_server_objective(args.address, args.port)

    ga_study = None
    de_study = None
    enqueue_baselines = not args.no_enqueue_baselines

    if args.method in {"ga", "both"}:
        print("=" * 72)
        print("Tuning GA")
        ga_study = tune_ga(
            objective=objective,
            seeds=seeds,
            budget=args.budget,
            n_trials=args.n_trials,
            optuna_seed=args.optuna_seed,
            enqueue_baselines=enqueue_baselines,
        )
        ga_dir = args.out / "ga"
        save_trials(ga_study, ga_dir / "trials.csv")
        save_best_config("ga", ga_study, ga_dir / "best_config.md", args.budget, seeds, args.n_trials)
        save_summaries("ga", ga_study, ga_dir)
        save_plots("ga", ga_study, ga_dir)

        print(f"Best GA params: {ga_study.best_params}")
        print(f"Best GA median: {ga_study.best_value}")

    if args.method in {"de", "both"}:
        print("=" * 72)
        print("Tuning DE")
        de_study = tune_de(
            objective=objective,
            seeds=seeds,
            budget=args.budget,
            n_trials=args.n_trials,
            optuna_seed=args.optuna_seed + 1,
            enqueue_baselines=enqueue_baselines,
        )
        de_dir = args.out / "de"
        save_trials(de_study, de_dir / "trials.csv")
        save_best_config("de", de_study, de_dir / "best_config.md", args.budget, seeds, args.n_trials)
        save_summaries("de", de_study, de_dir)
        save_plots("de", de_study, de_dir)

        print(f"Best DE params: {de_study.best_params}")
        print(f"Best DE median: {de_study.best_value}")

    if ga_study is not None and de_study is not None:
        save_selected_summary(ga_study, de_study, args.out / "selected_hyperparameters.csv")

    print()
    print("Created results in:")
    print(args.out)
    print()
    print("Important files:")
    if ga_study is not None:
        print(f"- {args.out / 'ga' / 'trials.csv'}")
        print(f"- {args.out / 'ga' / 'best_config.md'}")
        print(f"- {args.out / 'ga' / 'plots'}")
        print(f"- {args.out / 'ga' / 'summaries' / 'crossover_summary.csv'}")
    if de_study is not None:
        print(f"- {args.out / 'de' / 'trials.csv'}")
        print(f"- {args.out / 'de' / 'best_config.md'}")
        print(f"- {args.out / 'de' / 'plots'}")
    if ga_study is not None and de_study is not None:
        print(f"- {args.out / 'selected_hyperparameters.csv'}")


if __name__ == "__main__":
    main()
