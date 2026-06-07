import numpy as np
from evaluate import merit_function
from vecinos import rail_interval_for_point


# ============================================================
#  REPARACIÓN DEL VECTOR
#  x = (pA_x, pA_y, pA_z, pB_x, pB_y, pB_z, sA, sB, sRail, pRail)
# ============================================================

def repair_vector(x, rail_min, rail_max):
    x[6] = np.clip(x[6], 0.0, 1.0)  # sA
    x[7] = np.clip(x[7], 0.0, 1.0)  # sB
    x[8] = np.clip(x[8], 0.0, 1.0)  # sRail
    x[9] = np.clip(x[9], rail_min, rail_max)  # pRail
    return x


# ============================================================
#  ALGORITMO DE EVOLUCIÓN DIFERENCIAL (DE/rand/1/bin)
#  AHORA SOBRE POSICIONES + TIMING
# ============================================================

def de_timing_level(
    sol_init,
    pop_size=60,
    generations=150,
    F=0.75,
    CR=0.90,
    store_history=True
):
    """
    Optimización DE/rand/1/bin sobre:
        x = (pA_x, pA_y, pA_z,
             pB_x, pB_y, pB_z,
             sA, sB, sRail, pRail)

    Usa sol_init como punto factible inicial (Nivel 1).
    """

    rail_min = sol_init.rail_min
    rail_max = sol_init.rail_max

    x0 = np.zeros(10)
    x0[0:3] = sol_init.pA
    x0[3:6] = sol_init.pB
    x0[6] = sol_init.sA
    x0[7] = sol_init.sB
    x0[8] = sol_init.sRail
    x0[9] = sol_init.pRail

    pop = np.zeros((pop_size, 10))
    for i in range(pop_size):
        pop[i] = x0.copy()
        pop[i][0:3] += np.random.normal(0, 0.05, size=3)
        pop[i][3:6] += np.random.normal(0, 0.05, size=3)
        pop[i][6] = np.random.uniform(0.0, 1.0)
        pop[i][7] = np.random.uniform(0.0, 1.0)
        pop[i][8] = np.random.uniform(0.0, 1.0)
        pop[i][9] = np.random.uniform(rail_min, rail_max)

    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        sol = sol_init.copy()

        sol.pA = pop[i][0:3]
        sol.pB = pop[i][3:6]
        sol.sA = pop[i][6]
        sol.sB = pop[i][7]
        sol.sRail = pop[i][8]
        sol.pRail = pop[i][9]

        ok, rmin, rmax = rail_interval_for_point(sol.pB)
        if not ok:
            fitness[i] = 1e6
            continue

        sol.rail_min = rmin
        sol.rail_max = rmax
        sol.pRail = np.clip(sol.pRail, rmin, rmax)

        J, _, _ = merit_function(sol)
        fitness[i] = J

    history_J = [np.min(fitness)]

    for gen in range(generations):
        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = np.random.choice(idxs, size=3, replace=False)

            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = repair_vector(mutant, rail_min, rail_max)

            mask = np.random.rand(10) < CR
            mask[np.random.randint(10)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = repair_vector(trial, rail_min, rail_max)

            sol = sol_init.copy()
            sol.pA = trial[0:3]
            sol.pB = trial[3:6]
            sol.sA = trial[6]
            sol.sB = trial[7]
            sol.sRail = trial[8]
            sol.pRail = trial[9]

            ok, rmin, rmax = rail_interval_for_point(sol.pB)
            if not ok:
                J_trial = 1e6
            else:
                sol.rail_min = rmin
                sol.rail_max = rmax
                sol.pRail = np.clip(sol.pRail, rmin, rmax)
                J_trial, _, _ = merit_function(sol)

            if J_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = J_trial

        pop = new_pop
        fitness = new_fit
        history_J.append(np.min(fitness))

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]

    best_sol = sol_init.copy()
    best_sol.pA = best_x[0:3]
    best_sol.pB = best_x[3:6]
    best_sol.sA = best_x[6]
    best_sol.sB = best_x[7]
    best_sol.sRail = best_x[8]
    best_sol.pRail = best_x[9]

    ok, rmin, rmax = rail_interval_for_point(best_sol.pB)
    if ok:
        best_sol.rail_min = rmin
        best_sol.rail_max = rmax
        best_sol.pRail = np.clip(best_sol.pRail, rmin, rmax)

    J_best, tA_best, tB_best = merit_function(best_sol)

    return best_sol, J_best, best_idx, history_J
