import numpy as np
from evaluate import merit_function


# ============================================================
#  REPARACIÓN DEL VECTOR (solo pRail)
# ============================================================

def repair_vector(x, rail_min, rail_max):
    """
    Repara el vector x = (sA, sB, sRail, pRail)
    - Velocidades en [0,1]
    - pRail en [rail_min, rail_max]
    """
    x[0] = np.clip(x[0], 0.0, 1.0)
    x[1] = np.clip(x[1], 0.0, 1.0)
    x[2] = np.clip(x[2], 0.0, 1.0)
    x[3] = np.clip(x[3], rail_min, rail_max)
    return x


# ============================================================
#  ALGORITMO DE EVOLUCIÓN DIFERENCIAL (DE/rand/1/bin)
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
        x = (sA, sB, sRail, pRail)

    Mantiene sol_init.pA, sol_init.pB, sol_init.rail_min, sol_init.rail_max.
    """

    # ------------------------------------------------------------
    # 1. Preparación
    # ------------------------------------------------------------
    rail_min = sol_init.rail_min
    rail_max = sol_init.rail_max

    # Vector inicial
    x0 = np.array([sol_init.sA, sol_init.sB, sol_init.sRail, sol_init.pRail])

    # Inicialización uniforme en (0,1] excepto pRail que se escala
    pop = np.zeros((pop_size, 4))
    for i in range(pop_size):
        pop[i, 0] = np.random.uniform(0.0, 1.0)
        pop[i, 1] = np.random.uniform(0.0, 1.0)
        pop[i, 2] = np.random.uniform(0.0, 1.0)
        pop[i, 3] = np.random.uniform(rail_min, rail_max)

    # Evaluación inicial
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        sol = sol_init.copy()
        sol.sA, sol.sB, sol.sRail, sol.pRail = pop[i]
        J, _, _ = merit_function(sol)
        fitness[i] = J

    # Historial
    history_J = [np.min(fitness)]

    # ------------------------------------------------------------
    # 2. Bucle evolutivo
    # ------------------------------------------------------------
    for gen in range(generations):

        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in range(pop_size):

            # --- Selección de r1, r2, r3 ---
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = np.random.choice(idxs, size=3, replace=False)

            # --- Mutación ---
            mutant = pop[r1] + F * (pop[r2] - pop[r3])

            # --- Reparación del mutante ---
            mutant = repair_vector(mutant, rail_min, rail_max)

            # --- Cruce binomial ---
            mask = np.random.rand(4) < CR
            mask[np.random.randint(4)] = True  # asegurar al menos un gen
            trial = np.where(mask, mutant, pop[i])

            # --- Reparación del trial ---
            trial = repair_vector(trial, rail_min, rail_max)

            # --- Evaluación ---
            sol = sol_init.copy()
            sol.sA, sol.sB, sol.sRail, sol.pRail = trial
            J_trial, _, _ = merit_function(sol)

            # --- Selección elitista ---
            if J_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = J_trial

        pop = new_pop
        fitness = new_fit

        history_J.append(np.min(fitness))

    # ------------------------------------------------------------
    # 3. Mejor individuo final
    # ------------------------------------------------------------
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]

    best_sol = sol_init.copy()
    best_sol.sA, best_sol.sB, best_sol.sRail, best_sol.pRail = best_x

    J_best, tA_best, tB_best = merit_function(best_sol)

    return best_sol, J_best, best_idx, history_J
