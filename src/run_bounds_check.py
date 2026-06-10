import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def patch_bounds(lower: float, upper: float):
    # We patch bounds before importing algorithms.
    # misc.py copies LOWER_BOUND and UPPER_BOUND from common_types.py,
    # so we also patch misc after import.
    import common_types

    common_types.LOWER_BOUND = lower
    common_types.UPPER_BOUND = upper

    import misc

    misc.LOWER_BOUND = lower
    misc.UPPER_BOUND = upper

    return common_types


def read_cases(path: Path, Case):
    def as_int(value, default):
        if value in ("", None):
            return default
        return int(float(value))

    def as_float(value, default):
        if value in ("", None):
            return default
        return float(value)

    def as_text(value, default):
        if value in ("", None):
            return default
        return str(value)

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cases = []
    for row in rows:
        cases.append(
            Case(
                name=row["case"],
                method=row["method"],
                budget=as_int(row["budget"], 100),
                seed=as_int(row["seed"], 0),
                population=as_int(row.get("population"), 10),
                mutation_sigma=as_float(row.get("mutation_sigma"), 0.7),
                crossover=as_text(row.get("crossover"), "arithmetic"),
                differential_weight=as_float(row.get("differential_weight"), 0.7),
                crossover_rate=as_float(row.get("crossover_rate"), 0.9),
            )
        )

    return cases


def variant_name(case):
    method = case.method.lower()

    if method == "random":
        return "random"

    if method == "ga":
        return f"ga_{case.crossover}"

    if method in {"de", "de_rand_1_bin"}:
        return f"de_F{case.differential_weight:.2f}_CR{case.crossover_rate:.2f}"

    return method


def median(values):
    return statistics.median(values) if values else ""


def iqr(values):
    if len(values) < 4:
        return ""
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(case, objective):
    from random_search import random_search
    from genetic_algorithm import genetic_algorithm
    from differential_evolution import differential_evolution

    method = case.method.lower()

    if method == "random":
        return random_search(objective, min, case)

    if method == "ga":
        return genetic_algorithm(objective, min, case)

    if method in {"de", "de_rand_1_bin"}:
        return differential_evolution(objective, min, case)

    raise ValueError(f"Unknown method: {case.method}")


def eval_to_best(history, best_value, eps=1e-12):
    for evaluation, value in history:
        if abs(float(value) - float(best_value)) <= eps:
            return evaluation
    return ""


def make_summary(rows, key_fields):
    groups = {}

    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)

    summaries = []

    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        values = [float(row["best_value"]) for row in group_rows]
        evals = [int(row["eval_to_best"]) for row in group_rows if row["eval_to_best"] != ""]

        summary = {field: value for field, value in zip(key_fields, key)}
        summary.update(
            {
                "runs": len(group_rows),
                "best_value_min": min(values),
                "best_value_median": median(values),
                "best_value_iqr": iqr(values),
                "best_value_max": max(values),
                "eval_to_best_median": median(evals),
                "eval_to_best_iqr": iqr(evals),
            }
        )
        summaries.append(summary)

    return summaries


def make_plots(out_dir, run_rows, history_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"Skipping plots, matplotlib not available: {error}")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Convergence: one line per variant, averaged by evaluation index.
    grouped = {}
    for row in history_rows:
        key = (row["variant"], int(row["evaluation"]))
        grouped.setdefault(key, []).append(float(row["best_value"]))

    by_variant = {}
    for (variant, evaluation), values in grouped.items():
        by_variant.setdefault(variant, []).append((evaluation, statistics.mean(values)))

    plt.figure(figsize=(9, 5))
    for variant, points in sorted(by_variant.items()):
        points.sort()
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        plt.plot(xs, ys, label=variant)

    plt.xlabel("evaluation")
    plt.ylabel("best J(x)")
    plt.title("Bounds check convergence")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "convergence_by_variant.png", dpi=160)
    plt.close()

    # Boxplot: final best values per variant.
    variants = sorted({row["variant"] for row in run_rows})
    values = [[float(row["best_value"]) for row in run_rows if row["variant"] == variant] for variant in variants]

    plt.figure(figsize=(9, 5))
    plt.boxplot(values, labels=variants)
    plt.ylabel("best J(x)")
    plt.title("Final best values by variant")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "boxplot_by_variant.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run bounds sensitivity cases without editing common_types.py.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lower", type=float, default=-10.0)
    parser.add_argument("--upper", type=float, default=10.0)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    common_types = patch_bounds(args.lower, args.upper)

    from server_handler import make_server_objective

    print(f"Using bounds: [{common_types.LOWER_BOUND}, {common_types.UPPER_BOUND}]")
    print(f"Dimension: {common_types.DIMENSION}")

    objective = make_server_objective(args.address, args.port)
    cases = read_cases(args.cases, common_types.Case)

    run_rows = []
    history_rows = []

    for case in cases:
        variant = variant_name(case)

        print()
        print("=" * 72)
        print(f"Running case: {case.name}")
        print(f"method={case.method}, variant={variant}, budget={case.budget}, seed={case.seed}")
        print(f"bounds=[{args.lower}, {args.upper}]")

        start = time.time()
        best_x, best_value, history = run_case(case, objective)
        elapsed = time.time() - start

        reached_at = eval_to_best(history, best_value)

        run_rows.append(
            {
                "case": case.name,
                "method": case.method,
                "variant": variant,
                "budget": case.budget,
                "seed": case.seed,
                "lower_bound": args.lower,
                "upper_bound": args.upper,
                "population": case.population,
                "mutation_sigma": case.mutation_sigma if case.method == "ga" else "",
                "crossover": case.crossover if case.method == "ga" else "",
                "differential_weight": case.differential_weight if case.method != "ga" else "",
                "crossover_rate": case.crossover_rate if case.method != "ga" else "",
                "best_value": best_value,
                "eval_to_best": reached_at,
                "time_seconds": elapsed,
                "x1": best_x[0],
                "x2": best_x[1],
                "x3": best_x[2],
                "x4": best_x[3],
            }
        )

        for evaluation, value in history:
            history_rows.append(
                {
                    "case": case.name,
                    "method": case.method,
                    "variant": variant,
                    "budget": case.budget,
                    "seed": case.seed,
                    "evaluation": evaluation,
                    "best_value": value,
                }
            )

        print(f"DONE: best J(x) = {best_value:.12g}, eval_to_best={reached_at}, x={best_x}, time={elapsed:.2f}s")

    run_fields = [
        "case",
        "method",
        "variant",
        "budget",
        "seed",
        "lower_bound",
        "upper_bound",
        "population",
        "mutation_sigma",
        "crossover",
        "differential_weight",
        "crossover_rate",
        "best_value",
        "eval_to_best",
        "time_seconds",
        "x1",
        "x2",
        "x3",
        "x4",
    ]

    history_fields = [
        "case",
        "method",
        "variant",
        "budget",
        "seed",
        "evaluation",
        "best_value",
    ]

    summary_fields = [
        "variant",
        "budget",
        "runs",
        "best_value_min",
        "best_value_median",
        "best_value_iqr",
        "best_value_max",
        "eval_to_best_median",
        "eval_to_best_iqr",
    ]

    method_summary_fields = [
        "method",
        "budget",
        "runs",
        "best_value_min",
        "best_value_median",
        "best_value_iqr",
        "best_value_max",
        "eval_to_best_median",
        "eval_to_best_iqr",
    ]

    args.out.mkdir(parents=True, exist_ok=True)

    write_csv(args.out / "runs.csv", run_rows, run_fields)
    write_csv(args.out / "histories.csv", history_rows, history_fields)
    write_csv(args.out / "summary_by_variant.csv", make_summary(run_rows, ["variant", "budget"]), summary_fields)
    write_csv(args.out / "summary_by_method.csv", make_summary(run_rows, ["method", "budget"]), method_summary_fields)

    make_plots(args.out, run_rows, history_rows)

    print()
    print("Created:")
    print(args.out / "runs.csv")
    print(args.out / "histories.csv")
    print(args.out / "summary_by_variant.csv")
    print(args.out / "summary_by_method.csv")
    print(args.out / "plots" / "convergence_by_variant.png")
    print(args.out / "plots" / "boxplot_by_variant.png")


if __name__ == "__main__":
    main()
