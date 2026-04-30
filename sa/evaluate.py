import numpy as np
from reachability import is_reachable_world

BIG_PENALTY = 1e4

def evaluate(sol, robotA, robotB,
             rail_min=0.0, rail_max=0.70,
             stats=None):

    penalty = 0.0

    # --- 1. Alcanzabilidad geométrica + IK ---
    reachableA, dataA = is_reachable_world(robotA, sol.pA)
    reachableB, dataB = is_reachable_world(robotB, sol.pB)

    if stats is not None:
        if reachableA and reachableB:
            stats["feasible"] += 1

    if not reachableA or not reachableB:
        return BIG_PENALTY

    # --- 2. Separación ---
    desired_sep = 0.20
    sep = np.linalg.norm(sol.pB - sol.pA)
    penalty += 50.0 * (sep - desired_sep)**2

    # --- 3. Tiempos aproximados ---
    dA = np.linalg.norm(sol.pA - robotA.base.t)
    dB = np.linalg.norm(sol.pB - robotB.base.t)

    vA = max(sol.sA, 1e-3)
    vB = max(sol.sB, 1e-3)

    tA = dA / vA
    tB = dB / vB

    Tmax = max(tA, tB)

    penalty += 10.0 * (tA - tB)**2

    return Tmax + penalty
