import numpy as np
from dataclasses import dataclass
from evaluate import (
    DIM,
    LOWER,
    UPPER,
    RAIL_MIN,
    RAIL_MAX,
    objective,
    objective_components,
    is_feasible,
    repair_solution,
)


def sample_candidate(rng):
    for _ in range(5000):
        x = np.array([
            rng.uniform(0.0, 0.60),       # pAx
            rng.uniform(-0.50, 0.50),     # pAy
            rng.uniform(0.10, 0.90),      # pAz
            rng.uniform(0.20, 1.00),      # sA
            rng.uniform(0.20, 1.00),      # sB
            rng.uniform(0.20, 1.00),      # sRail
            rng.uniform(RAIL_MIN, RAIL_MAX)
        ])
        x = repair_solution(x)
        if is_feasible(x):
            return x
    return repair_solution(rng.uniform(LOWER, UPPER))


def initialize_population(rng, pop_size):
    return np.array([sample_candidate(rng) for _ in range(pop_size)])


@dataclass
class DEParams:
    pop_size: int = 60
    generations: int = 150
    F: float = 0.75
    CR: float = 0.90


def differential_evolution_rendezvous(params=DEParams(), seed=0, store_history=True):
    rng = np.random.default_rng(seed)
    pop = initialize_population(rng, params.pop_size)
    fitness = np.array([objective(ind) for ind in pop])

    history = {
        "best_J": [],
        "mean_J": [],
        "best_Tmax": [],
        "best_delta_t": [],
        "feasible_ratio": [],
        "populations": [],
    }

    for gen in range(params.generations):
        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in range(params.pop_size):
            candidates = [j for j in range(params.pop_size) if j != i]
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

            mutant = pop[r1] + params.F * (pop[r2] - pop[r3])
            mutant = repair_solution(mutant)

            mask = rng.random(DIM) < params.CR
            mask[rng.integers(DIM)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = repair_solution(trial)

            trial_fit = objective(trial)
            if trial_fit <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = trial_fit

        pop = new_pop
        fitness = new_fit

        best_idx = np.argmin(fitness)
        comp = objective_components(pop[best_idx])
        history["best_J"].append(comp["J"])
        history["mean_J"].append(np.mean(fitness))
        history["best_Tmax"].append(comp["Tmax"])
        history["best_delta_t"].append(comp["delta_t"])
        history["feasible_ratio"].append(
            np.mean([is_feasible(ind) for ind in pop])
        )

        if store_history:
            history["populations"].append(pop.copy())

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx]
    return best_x, objective_components(best_x), history


# --- wrapper con el nombre antiguo para no romper main.py ---

def de_timing_level(sol_init=None,
                    pop_size=60,
                    generations=150,
                    F=0.75,
                    CR=0.90,
                    store_history=True):
    params = DEParams(pop_size=pop_size, generations=generations, F=F, CR=CR)
    best_x, best_comp, history = differential_evolution_rendezvous(
        params=params,
        seed=0,
        store_history=store_history
    )
    # Para compatibilidad con el código que espera (sol, J_best, best_iter, history_J)
    J_best = best_comp["J"]
    best_iter = np.argmin(history["best_J"])
    history_J = history["best_J"]
    return best_x, J_best, best_iter, history_J
