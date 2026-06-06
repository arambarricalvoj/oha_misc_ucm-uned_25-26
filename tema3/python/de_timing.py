import numpy as np
from vecinos import generate_timing_neighbor
from evaluate import merit_function


class DEParamsTiming:
    def __init__(self,
                 pop_size=40,
                 generations=150,
                 F=0.75,
                 CR=0.90,
                 seed=None):
        self.pop_size = pop_size
        self.generations = generations
        self.F = F
        self.CR = CR
        self.seed = seed


def de_timing_level(sol_init, params=DEParamsTiming()):
    """
    Evolución Diferencial (DE/rand/1/bin) trabajando DIRECTAMENTE con objetos Solution.
    Misma estructura que sa_timing_level, sin modificar Solution ni main.
    """

    rng = np.random.default_rng(params.seed)

    # ============================================================
    # 1. Inicializar población de objetos Solution
    # ============================================================
    pop = []
    for _ in range(params.pop_size):
        ind = generate_timing_neighbor(sol_init)   # igual que SA
        pop.append(ind)

    # Evaluación inicial
    fitness = []
    for ind in pop:
        J, tA, tB = merit_function(ind)
        fitness.append(J)
    fitness = np.array(fitness)

    # Mejor global
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    J_best = fitness[best_idx]
    best_iter = 0

    history_J = [J_best]

    # ============================================================
    # 2. Bucle principal DE
    # ============================================================
    for gen in range(params.generations):

        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in range(params.pop_size):

            # Seleccionar r1, r2, r3 distintos de i
            idxs = [j for j in range(params.pop_size) if j != i]
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)

            x1 = pop[r1]
            x2 = pop[r2]
            x3 = pop[r3]

            # ====================================================
            # MUTACIÓN (pero usando vecinos, no vectores)
            # ====================================================
            # Creamos un "mutante" aplicando perturbaciones
            mutant = generate_timing_neighbor(x1)

            # ====================================================
            # CRUCE BINOMIAL
            # ====================================================
            # trial = mezcla entre pop[i] y mutant
            trial = pop[i].copy()

            # Para cada atributo, CR decide si copiar del mutante
            if rng.random() < params.CR:
                trial.pA = mutant.pA.copy()
            if rng.random() < params.CR:
                trial.pB = mutant.pB.copy()
            if rng.random() < params.CR:
                trial.qA = mutant.qA.copy()
            if rng.random() < params.CR:
                trial.qB = mutant.qB.copy()
            if rng.random() < params.CR:
                trial.sA = mutant.sA
            if rng.random() < params.CR:
                trial.sB = mutant.sB
            if rng.random() < params.CR:
                trial.sRail = mutant.sRail

            # ====================================================
            # EVALUACIÓN
            # ====================================================
            J_trial, tA_trial, tB_trial = merit_function(trial)

            # ====================================================
            # SELECCIÓN
            # ====================================================
            if J_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = J_trial

        # Actualizar población
        pop = new_pop
        fitness = new_fit

        # Mejor global
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < J_best:
            J_best = fitness[best_idx]
            best = pop[best_idx].copy()
            best_iter = gen

        history_J.append(J_best)

    return best, J_best, best_iter, history_J
