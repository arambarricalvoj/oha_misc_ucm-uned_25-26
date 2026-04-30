import numpy as np
from vecinos import generate_timing_neighbor
from evaluate import merit_function


def sa_timing_level(sol_init, max_iters=300,
                    T0=1.0, alpha=0.95):
    """
    SA de nivel 2:
    - Variables: sA, sB, sRail, pRail
    - Vecino: generate_timing_neighbor
    - Coste: función de mérito J_total (solo penalizaciones)
    """
    current = sol_init.copy()
    best = current.copy()

    J_current, _, _ = merit_function(current)
    J_best = J_current

    T = T0

    for k in range(max_iters):
        neighbor = generate_timing_neighbor(current)
        J_neighbor, _, _ = merit_function(neighbor)

        dJ = J_neighbor - J_current

        if dJ < 0:
            accept = True
        else:
            prob = np.exp(-dJ / (T + 1e-12))
            accept = (np.random.rand() < prob)

        if accept:
            current = neighbor
            J_current = J_neighbor

            if J_current < J_best:
                J_best = J_current
                best = current.copy()

        T *= alpha

    return best
