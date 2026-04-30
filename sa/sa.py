import numpy as np
from neighbor import generate_neighbor

def simulated_annealing(
        x0,
        evaluate,
        robotA,
        robotB,
        sigma_p,
        sigma_q,
        sigma_s,
        p_min,
        p_max,
        T0=1.0,
        alpha=0.95,
        L=50,
        K=200):

    stats = {
        "generated": 0,
        "accepted": 0,
        "feasible": 0,
        "history": []
    }

    x = x0
    fx = evaluate(x, robotA, robotB, stats=stats)

    best = x
    fbest = fx

    T = T0

    for k in range(K):
        for _ in range(L):

            stats["generated"] += 1

            x_new = generate_neighbor(
                x, robotA, robotB,
                sigma_p, sigma_q, sigma_s,
                p_min, p_max
            )

            f_new = evaluate(x_new, robotA, robotB, stats=stats)

            delta = f_new - fx
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                x = x_new
                fx = f_new
                stats["accepted"] += 1

                if f_new < fbest:
                    best = x_new
                    fbest = f_new

        stats["history"].append(fbest)
        T *= alpha

    return best, fbest, stats
