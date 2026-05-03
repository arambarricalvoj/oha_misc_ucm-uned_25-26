import numpy as np
from vecinos import generate_timing_neighbor
from evaluate import merit_function

def sa_timing_level(sol_init, max_iters=300,
                    T0=1.0, alpha=0.98):

    current = sol_init.copy()
    best = current.copy()

    J_current, tA_current, tB_current = merit_function(current)
    J_best = J_current
    best_iter = 0

    T = T0

    # historial para curvas de convergencia
    history_J = [J_current]

    for k in range(max_iters):

        neighbor = generate_timing_neighbor(current)
        J_neighbor, tA_n, tB_n = merit_function(neighbor)

        dJ = J_neighbor - J_current

        if dJ < 0:
            accept = True
        else:
            prob = np.exp(-dJ / (T + 1e-12))
            accept = (np.random.rand() < prob)

        if accept:
            current = neighbor
            J_current = J_neighbor
            tA_current = tA_n
            tB_current = tB_n

            if J_current < J_best:
                J_best = J_current
                best = current.copy()
                best_iter = k

        T *= alpha

        history_J.append(J_current)

    # devolver todo lo necesario
    return best, J_best, best_iter, history_J
