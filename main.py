import argparse
import math
from pathlib import Path
import sys
from typing import Callable
import xmlrpc.client

Vector = list[float]
Objective = Callable[[Vector], float]

BAD_VALUE = 1_000_000.0

def make_server_objective(address: str, port: int) -> Objective:
    client = xmlrpc.client.ServerProxy(f"http://{address}:{port}", allow_none=True)

    def objective(candidate: Vector) -> float:
        return float(client.evaluate([float(v) for v in candidate]))

    return objective

def check_server(objective: Objective) -> None:
    print("Checking server connection. Evaluationg example vectors:")
    for candidate in ([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [-4.0, -4.0, -4.0, 4.0]):
        value = safe_evaluate(objective, list(candidate))
        print(f"  J({list(candidate)}) = {value}")

def safe_evaluate(objective: Objective, candidate: Vector) -> float:
    try:
        value = float(objective(candidate))
        if not math.isfinite(value):
            return BAD_VALUE
        return value
    except Exception as error:
        print(f"  evaluation failed for {candidate}: {error}")
        return BAD_VALUE

def main() -> None:
    parser = argparse.ArgumentParser(description="Beta runner for WAE VSI experiments through the MATLAB XML-RPC server.")
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="Only check a few objective values and exit.")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    objective = make_server_objective(args.address, args.port)
    if args.check:
        check_server(objective)
        return

if __name__ == "__main__":
    main()