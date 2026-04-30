import numpy as np
from reachability import reachable_with_rail

def generate_neighbor(sol, robotA, robotB,
                      sigma_p, sigma_q, sigma_s,
                      p_min, p_max,
                      rail_min=0.0, rail_max=0.70,
                      max_attempts=20):

    for _ in range(max_attempts):
        new = sol.copy()

        # --- 1. Mutación de posición ---
        new.pA += np.random.normal(0, sigma_p, size=3)
        new.pA = np.clip(new.pA, p_min, p_max)

        # --- 2. Mutación de orientación ---
        new.qA += np.random.normal(0, sigma_q, size=4)
        new.qA /= np.linalg.norm(new.qA)

        # --- 3. Mutación de velocidades ---
        new.sA = np.clip(new.sA + np.random.normal(0, sigma_s), 0, 1)
        new.sB = np.clip(new.sB + np.random.normal(0, sigma_s), 0, 1)
        new.sRail = np.clip(new.sRail + np.random.normal(0, sigma_s), 0, 1)

        # Recalcular dependientes
        new.update_dependent()

        # --- 4. Comprobación de workspace geométrico ---
        s_world = rail_min + new.sRail * (rail_max - rail_min)

        if reachable_with_rail(robotA, new.pA, 0) and \
           reachable_with_rail(robotB, new.pA, s_world):
            return new

    # Si no se encuentra vecino válido, devolver el original
    return sol.copy()
