import argparse
from dataclasses import dataclass
import time

from common_types import Case, Objective
from random_search import random_search
from server_handler import check_server, make_server_objective
    
def run_cases(objective: Objective, cases: list[Case]) -> None:
    for case in cases:
        print("\n" + "=" * 72)
        print(f"Running case: {case.name}")
        print(f"method={case.method}, budget={case.budget}, seed={case.seed}")
        start = time.perf_counter()

        if case.method == "random":
            best_candidate, best_value, history = random_search(objective, case)
        else:
            raise ValueError(f"Unknown method: {case.method}")

        elapsed = time.perf_counter() - start
        print(f"DONE: best J(x) = {best_value:.12g}, x = {best_candidate}, time = {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Beta runner for WAE VSI experiments through the MATLAB XML-RPC server.")
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="Only check a few objective values and exit.")
    args = parser.parse_args()

    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)

    objective = make_server_objective(args.address, args.port)
    if args.check:
        check_server(objective)
        return
    
    run_cases(objective, [Case("test", "random", 10, 10)])

if __name__ == "__main__":
    main()