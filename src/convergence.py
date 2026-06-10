import argparse
import csv
import statistics
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def median(values):
    return statistics.median(values) if values else ""


def iqr(values):
    if len(values) < 4:
        return ""
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    runs = read_csv(args.results / "runs.csv")
    histories = read_csv(args.results / "histories.csv")

    history_by_run = {}
    for row in histories:
        key = (row["case"], row["method"], row["seed"])
        history_by_run.setdefault(key, []).append(row)

    per_run_rows = []

    for run in runs:
        key = (run["case"], run["method"], run["seed"])
        points = history_by_run.get(key, [])

        final_best = float(run["best_value"])
        budget = int(run["budget"])

        eval_to_best = ""
        for point in points:
            value = float(point["best_value"])
            if abs(value - final_best) < 1e-12:
                eval_to_best = int(point["evaluation"])
                break

        ratio = ""
        if eval_to_best != "":
            ratio = eval_to_best / budget

        per_run_rows.append({
            "case": run["case"],
            "method": run["method"],
            "budget": run["budget"],
            "seed": run["seed"],
            "best_value": run["best_value"],
            "eval_to_best": eval_to_best,
            "eval_to_best_ratio": ratio,
            "x1": run.get("x1", ""),
            "x2": run.get("x2", ""),
            "x3": run.get("x3", ""),
            "x4": run.get("x4", ""),
        })

    write_csv(
        args.results / "convergence_summary.csv",
        per_run_rows,
        [
            "case",
            "method",
            "budget",
            "seed",
            "best_value",
            "eval_to_best",
            "eval_to_best_ratio",
            "x1",
            "x2",
            "x3",
            "x4",
        ],
    )

    groups = {}
    for row in per_run_rows:
        key = (row["method"], row["budget"])
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for (method, budget), rows in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        best_values = [float(r["best_value"]) for r in rows]
        evals = [int(r["eval_to_best"]) for r in rows if r["eval_to_best"] != ""]
        ratios = [float(r["eval_to_best_ratio"]) for r in rows if r["eval_to_best_ratio"] != ""]

        aggregate_rows.append({
            "method": method,
            "budget": budget,
            "runs": len(rows),
            "best_value_min": min(best_values),
            "best_value_median": median(best_values),
            "best_value_iqr": iqr(best_values),
            "best_value_max": max(best_values),
            "eval_to_best_median": median(evals),
            "eval_to_best_iqr": iqr(evals),
            "eval_to_best_ratio_median": median(ratios),
        })

    write_csv(
        args.results / "convergence_by_method_budget.csv",
        aggregate_rows,
        [
            "method",
            "budget",
            "runs",
            "best_value_min",
            "best_value_median",
            "best_value_iqr",
            "best_value_max",
            "eval_to_best_median",
            "eval_to_best_iqr",
            "eval_to_best_ratio_median",
        ],
    )

    print("Created:")
    print(args.results / "convergence_summary.csv")
    print(args.results / "convergence_by_method_budget.csv")


if __name__ == "__main__":
    main()