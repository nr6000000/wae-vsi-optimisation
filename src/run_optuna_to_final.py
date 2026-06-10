import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print()
    print(">", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def read_best_trial(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing Optuna trials file: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    good_rows = []
    for row in rows:
        value = row.get("value_median", "")
        if value not in ("", None):
            try:
                row["_value"] = float(value)
                good_rows.append(row)
            except ValueError:
                pass

    if not good_rows:
        raise RuntimeError(f"No completed Optuna trials found in {path}")

    return min(good_rows, key=lambda r: r["_value"])


def nice_float(value, digits=2):
    # Keep values simple in final CSV.
    return f"{float(value):.{digits}f}"


def update_final_cases(template_path: Path, output_path: Path, ga_best, de_best):
    if not template_path.exists():
        raise FileNotFoundError(f"Missing final template CSV: {template_path}")

    ga_population = str(int(float(ga_best["population"])))
    ga_sigma = nice_float(ga_best["mutation_sigma"])
    ga_crossover = ga_best["crossover"]

    de_population = str(int(float(de_best["population"])))
    de_f = nice_float(de_best["differential_weight"])
    de_cr = nice_float(de_best["crossover_rate"])

    with template_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError(f"CSV has no header: {template_path}")

        rows = []
        for row in reader:
            method = row["method"].strip().lower()

            if method == "ga":
                row["population"] = ga_population
                row["mutation_sigma"] = ga_sigma
                row["crossover"] = ga_crossover
                row["differential_weight"] = ""
                row["crossover_rate"] = ""

            elif method in {"de", "de_rand_1_bin"}:
                row["population"] = de_population
                row["mutation_sigma"] = ""
                row["crossover"] = ""
                row["differential_weight"] = de_f
                row["crossover_rate"] = de_cr

            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Selected final parameters:")
    print(f"  GA: population={ga_population}, mutation_sigma={ga_sigma}, crossover={ga_crossover}")
    print(f"  DE: population={de_population}, F={de_f}, CR={de_cr}")
    print(f"  written to: {output_path}")


def print_final_info(final_out: Path):
    print()
    print("Final result files:")
    print(f"  {final_out / 'summary.csv'}")
    print(f"  {final_out / 'runs.csv'}")
    print(f"  {final_out / 'histories.csv'}")
    print(f"  {final_out / 'plots' / 'convergence.png'}")
    print(f"  {final_out / 'plots' / 'boxplot.png'}")
    print()
    print("Useful PowerShell commands:")
    print(f"  Start-Process .\\{final_out}\\summary.csv")
    print(f"  Start-Process .\\{final_out}\\runs.csv")
    print(f"  Start-Process .\\{final_out}\\plots\\convergence.png")
    print(f"  Start-Process .\\{final_out}\\plots\\boxplot.png")


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna tuning, copy best parameters to final CSV, then run final comparison."
    )

    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)

    parser.add_argument("--optuna-budget", type=int, default=500)
    parser.add_argument("--optuna-seeds", default="316018,348556,1001")
    parser.add_argument("--optuna-trials", type=int, default=12)
    parser.add_argument("--optuna-out", type=Path, default=Path("results/optuna_tuning"))

    parser.add_argument(
        "--final-template",
        type=Path,
        default=Path("cases/final_comparison_10seeds_simple.csv"),
    )
    parser.add_argument(
        "--final-cases",
        type=Path,
        default=Path("cases/final_comparison_10seeds_optuna.csv"),
    )
    parser.add_argument(
        "--final-out",
        type=Path,
        default=Path("results/final_10seeds_optuna"),
    )

    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--skip-final", action="store_true")

    args = parser.parse_args()
    python = sys.executable

    if not args.skip_check:
        run([
            python, "src/main.py",
            "--check",
            "--address", args.address,
            "--port", str(args.port),
        ])

    if not args.skip_optuna:
        run([
            python, "src/optuna_tuning.py",
            "--method", "both",
            "--budget", str(args.optuna_budget),
            "--seeds", args.optuna_seeds,
            "--n-trials", str(args.optuna_trials),
            "--address", args.address,
            "--port", str(args.port),
            "--out", str(args.optuna_out),
        ])

    ga_best = read_best_trial(args.optuna_out / "ga" / "trials.csv")
    de_best = read_best_trial(args.optuna_out / "de" / "trials.csv")
    update_final_cases(args.final_template, args.final_cases, ga_best, de_best)

    if not args.skip_final:
        run([
            python, "src/main.py",
            "--cases", str(args.final_cases),
            "--out", str(args.final_out),
            "--address", args.address,
            "--port", str(args.port),
        ])

    print_final_info(args.final_out)


if __name__ == "__main__":
    main()
