import numpy as np
from vecinos import generate_timing_neighbor
from evaluate import merit_function


def sa_timing_level(sol_init, max_iters=300,
                    T0=1.0, alpha=0.95):

    current = sol_init.copy()
    best = current.copy()

    J_current, tA_current, tB_current = merit_function(current)
    J_best = J_current

    best_iter = 0   # <-- para registrar en qué iteración se logró el mejor J

    T = T0

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

            # Registrar mejor solución
            if J_current < J_best:
                J_best = J_current
                best = current.copy()
                best_iter = k   # <-- guardamos la iteración exacta

        T *= alpha

        # Imprimir evolución
        if k % 50 == 0:
            print(f"[Iter {k}] J={J_current:.4f}, "
                    f"tA={tA_current:.3f}, tB={tB_current:.3f}, "
                    f"sA={current.sA:.2f}, sB={current.sB:.2f}, "
                    f"sRail={current.sRail:.2f}, pRail={current.pRail:.3f}")

    # Al final, imprimir la mejor iteración
    print("\n>>> Mejor solución encontrada en iteración", best_iter)
    print(f"    J_best={J_best:.4f}")
    print(f"    tA_best={tA_current:.3f}, tB_best={tB_current:.3f}")
    print(f"    sA={best.sA:.2f}, sB={best.sB:.2f}, "
          f"sRail={best.sRail:.2f}, pRail={best.pRail:.3f}")

    return best
