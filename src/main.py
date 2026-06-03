import argparse
import csv
import math
from pathlib import Path
from statistics import median
import statistics
import sys
import time
import matplotlib.pyplot as plt

from common_types import Case, Objective
from genetic_algorithm import genetic_algorithm
from random_search import random_search
from server_handler import check_server, make_server_objective
from differential_evolution import differential_evolution
    
def make_summary(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in run_rows:
        key = str(row["method"])
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for method, rows in sorted(groups.items()):
        values = [float(row["best_value"]) for row in rows]
        summary_rows.append(
            {
                "method": method,
                "runs": len(rows),
                "best_value_min": min(values),
                "best_value_median": median(values),
                "best_value_iqr": iqr(values),
                "best_value_max": max(values),
            }
        )
    return summary_rows

def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def make_plots(out_dir: Path, history_rows: list[dict[str, object]], run_rows: list[dict[str, object]]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot every run as a light, simple line. This is beta, not a final paper figure.
    plt.figure(figsize=(8, 5))
    by_run: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    for row in history_rows:
        key = (str(row["case"]), str(row["method"]), str(row["seed"]))
        by_run.setdefault(key, []).append((int(row["evaluation"]), float(row["best_value"])))
    for (case_name, method, seed), points in by_run.items():
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, label=f"{method}, seed={seed}")
    plt.xlabel("number of simulator evaluations")
    plt.ylabel("best J(x) found so far")
    plt.title("WAE VSI beta: convergence")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "convergence.png", dpi=160)
    plt.close()

    by_method: dict[str, list[float]] = {}
    for row in run_rows:
        by_method.setdefault(str(row["method"]), []).append(float(row["best_value"]))
    labels = sorted(by_method)
    data = [by_method[label] for label in labels]
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel("final best J(x)")
    plt.title("WAE VSI beta: final values")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "boxplot.png", dpi=160)
    plt.close()

def iqr(values: list[float]) -> float:
    if len(values) < 4:
        return math.nan
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]

def run_cases(objective: Objective, comparator, cases: list[Case], out_dir: Path) -> None:
    run_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    
    for case in cases:
        print("\n" + "=" * 72)
        print(f"Running case: {case.name}")
        print(f"method={case.method}, budget={case.budget}, seed={case.seed}")
        start = time.perf_counter()

        if case.method == "random":
            best_candidate, best_value, history = random_search(objective, comparator, case)
        elif case.method == "ga":
            best_candidate, best_value, history = genetic_algorithm(objective, comparator, case)
        elif case.method in {"de", "de_rand_1_bin"}:
            best_candidate, best_value, history = differential_evolution(objective, comparator, case)
        else:
            raise ValueError(f"Unknown method: {case.method}")

        elapsed = time.perf_counter() - start
        print(f"DONE: best J(x) = {best_value:.12g}, x = {best_candidate}, time = {elapsed:.2f}s")

        run_row: dict[str, object] = {
            "case": case.name,
            "method": case.method,
            "budget": case.budget,
            "seed": case.seed,
            "population": case.population,
            "mutation_sigma": case.mutation_sigma,
            "crossover": case.crossover,
            "differential_weight": case.differential_weight,
            "crossover_rate": case.crossover_rate,
            "best_value": best_value,
            "elapsed_sec": elapsed,
        }
        for i, value in enumerate(best_candidate, start=1):
            run_row[f"x{i}"] = value
        run_rows.append(run_row)

        for evaluation, value in history:
            history_rows.append(
                {
                    "case": case.name,
                    "method": case.method,
                    "seed": case.seed,
                    "evaluation": evaluation,
                    "best_value": value,
                }
            )

        summary_rows = make_summary(run_rows)

    write_csv(
        out_dir / "runs.csv",
        run_rows,
        [
            "case", "method", "budget", "seed", "population", "mutation_sigma", "crossover",
            "differential_weight", "crossover_rate", "best_value", "x1", "x2", "x3", "x4", "elapsed_sec",
        ],
    )
    write_csv(out_dir / "histories.csv", history_rows, ["case", "method", "seed", "evaluation", "best_value"])
    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        ["method", "runs", "best_value_min", "best_value_median", "best_value_iqr", "best_value_max"],
    )
    make_plots(out_dir, history_rows, run_rows)

    print("\nCreated:")
    print(f"  {out_dir / 'summary.csv'}")
    print(f"  {out_dir / 'runs.csv'}")
    print(f"  {out_dir / 'histories.csv'}")
    print(f"  {out_dir / 'plots' / 'convergence.png'}")
    print(f"  {out_dir / 'plots' / 'boxplot.png'}")


def parse_int(text: str, default: int) -> int:
    return int(text) if text.strip() else default

def parse_float(text: str, default: float) -> float:
    return float(text) if text.strip() else default

def read_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cases.append(
                Case(
                    name=row["case"].strip(),
                    method=row["method"].strip().lower(),
                    budget=parse_int(row.get("budget", ""), 40),
                    seed=parse_int(row.get("seed", ""), 316018),
                    population=parse_int(row.get("population", ""), 10),
                    mutation_sigma=parse_float(row.get("mutation_sigma", ""), 0.7),
                    crossover=(row.get("crossover", "") or "arithmetic").strip(),
                    differential_weight=parse_float(row.get("differential_weight", ""), 0.7),
                    crossover_rate=parse_float(row.get("crossover_rate", ""), 0.9),
                )
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Beta runner for WAE VSI experiments through the MATLAB XML-RPC server.")
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="Only check a few objective values and exit.")
    parser.add_argument("--cases", type=Path, default=Path("cases_tiny.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/tiny"))
    parser.add_argument("--max", action="store_true", help="Maximize objective instead of minimizing it.")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    objective = make_server_objective(args.address, args.port)
    if args.check:
        check_server(objective)
        return
    
    cases = read_cases(args.cases)

    comparator = max if args.max else min
    run_cases(objective, comparator, cases, args.out)

if __name__ == "__main__":
    main()