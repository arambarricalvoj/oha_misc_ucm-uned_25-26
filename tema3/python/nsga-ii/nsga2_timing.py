import numpy as np
from evaluate import compute_times


# ============================================================
# REPARACIÓN DEL VECTOR
# ============================================================

def repair_vector(x, rail_min, rail_max):
    """
    x = (sA, sB, sRail, pRail)
    """

    x[0] = np.clip(x[0], 0.05, 1.0)
    x[1] = np.clip(x[1], 0.05, 1.0)
    x[2] = np.clip(x[2], 0.05, 1.0)
    x[3] = np.clip(x[3], rail_min, rail_max)

    return x


# ============================================================
# DOMINANCIA DE PARETO
# ============================================================

def dominates(a, b):
    """
    a domina a b si:
      - no es peor en ningún objetivo
      - es mejor en al menos uno
    """

    return np.all(a <= b) and np.any(a < b)


# ============================================================
# FAST NON-DOMINATED SORT
# ============================================================

def fast_nondominated_sort(fitness):

    pop_size = len(fitness)

    S = [[] for _ in range(pop_size)]
    n = np.zeros(pop_size, dtype=int)

    fronts = [[]]

    for p in range(pop_size):

        for q in range(pop_size):

            if p == q:
                continue

            if dominates(fitness[p], fitness[q]):
                S[p].append(q)

            elif dominates(fitness[q], fitness[p]):
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    i = 0

    while len(fronts[i]) > 0:

        next_front = []

        for p in fronts[i]:

            for q in S[p]:

                n[q] -= 1

                if n[q] == 0:
                    next_front.append(q)

        i += 1
        fronts.append(next_front)

    return fronts[:-1]


# ============================================================
# CROWDING DISTANCE
# ============================================================

def crowding_distance(fitness, front):

    n = len(front)

    if n == 0:
        return np.array([])

    if n <= 2:
        return np.full(n, np.inf)

    distance = np.zeros(n)

    n_obj = fitness.shape[1]

    for m in range(n_obj):

        values = fitness[front, m]

        order = np.argsort(values)

        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf

        f_min = values[order[0]]
        f_max = values[order[-1]]

        if abs(f_max - f_min) < 1e-12:
            continue

        for i in range(1, n - 1):

            distance[order[i]] += (
                values[order[i + 1]]
                - values[order[i - 1]]
            ) / (f_max - f_min)

    return distance


# ============================================================
# NSGA-II + OPERADORES DE
# ============================================================

def nsga2_timing_level(
    sol_init,
    pop_size=60,
    generations=150,
    F=0.9,
    CR=0.9
):

    rail_min = sol_init.rail_min
    rail_max = sol_init.rail_max

    # ========================================================
    # POBLACIÓN INICIAL
    # ========================================================

    pop = np.zeros((pop_size, 4))

    for i in range(pop_size):

        pop[i, 0] = np.random.uniform(0.05, 1.0)   # sA
        pop[i, 1] = np.random.uniform(0.05, 1.0)   # sB
        pop[i, 2] = np.random.uniform(0.05, 1.0)   # sRail
        pop[i, 3] = np.random.uniform(rail_min, rail_max)

    # ========================================================
    # BUCLE EVOLUTIVO
    # ========================================================

    for gen in range(generations):

        offspring = np.zeros_like(pop)

        # ----------------------------------------------------
        # GENERAR DESCENDENCIA (DE/rand/1/bin)
        # ----------------------------------------------------

        for i in range(pop_size):

            idxs = [j for j in range(pop_size) if j != i]

            r1, r2, r3 = np.random.choice(idxs, size=3, replace=False)

            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = repair_vector(mutant, rail_min, rail_max)

            mask = np.random.rand(4) < CR
            mask[np.random.randint(4)] = True

            trial = np.where(mask, mutant, pop[i])
            trial = repair_vector(trial, rail_min, rail_max)

            offspring[i] = trial

        # ----------------------------------------------------
        # UNIÓN PADRES + HIJOS
        # ----------------------------------------------------

        combined = np.vstack((pop, offspring))

        # NUEVA FUNCIÓN OBJETIVO: (Tmax, |tA - tB|)
        fitness = np.zeros((2 * pop_size, 2))

        for i in range(2 * pop_size):

            sol = sol_init.copy()

            sol.sA = combined[i, 0]
            sol.sB = combined[i, 1]
            sol.sRail = combined[i, 2]
            sol.pRail = combined[i, 3]

            tA, tB = compute_times(sol)

            Tmax = max(tA, tB)
            deltaT = abs(tA - tB)

            fitness[i, 0] = Tmax
            fitness[i, 1] = deltaT

        # LOGGING
        if gen == 0:
            log = []

        gen_data = []
        for i in range(2 * pop_size):
            gen_data.append([
                combined[i,0], combined[i,1], combined[i,2], combined[i,3],
                fitness[i,0], fitness[i,1]
            ])
        log.append(gen_data)

        # ----------------------------------------------------
        # FAST NON-DOMINATED SORT
        # ----------------------------------------------------

        fronts = fast_nondominated_sort(fitness)

        selected = []

        for front in fronts:

            if len(selected) + len(front) <= pop_size:
                selected.extend(front)

            else:
                distances = crowding_distance(fitness, front)
                order = np.argsort(-distances)
                remaining = pop_size - len(selected)
                selected.extend([front[i] for i in order[:remaining]])
                break

        pop = combined[selected]

    # ========================================================
    # EVALUACIÓN FINAL
    # ========================================================

    fitness = np.zeros((pop_size, 2))

    for i in range(pop_size):

        sol = sol_init.copy()

        sol.sA = pop[i, 0]
        sol.sB = pop[i, 1]
        sol.sRail = pop[i, 2]
        sol.pRail = pop[i, 3]

        tA, tB = compute_times(sol)

        fitness[i, 0] = max(tA, tB)
        fitness[i, 1] = abs(tA - tB)

    fronts = fast_nondominated_sort(fitness)

    pareto_front = []

    for idx in fronts[0]:

        sol = sol_init.copy()

        sol.sA = pop[idx, 0]
        sol.sB = pop[idx, 1]
        sol.sRail = pop[idx, 2]
        sol.pRail = pop[idx, 3]

        pareto_front.append(sol)

    return pareto_front, log
