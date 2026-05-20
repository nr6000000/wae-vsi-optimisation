import math
import xmlrpc.client

from common_types import BAD_VALUE, Objective, Vector


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
