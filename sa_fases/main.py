import numpy as np
from solution import Solution
import vecinos
from sa_timing import sa_timing_level
from evaluate import evaluate_global_cost


def main():
    N_SAMPLES = 30
    offset = np.array([0.20, 0.0, 0.0])

    p_min = np.array([-1.0, -1.0, 0.0])
    p_max = np.array([ 1.5,  1.0, 1.0])

    best_global = None
    best_Tmax = np.inf
    best_J = np.inf

    print("\n=== NIVEL 1 + NIVEL 2 SOBRE 30 PUNTOS ===")

    for i in range(N_SAMPLES):
        print(f"\n--- Punto geométrico {i} ---")

        sol = Solution()
        sol = vecinos.generate_position_neighbor(sol, p_min, p_max, offset)

        print("Nivel 1:", sol)

        sol2 = sa_timing_level(sol, max_iters=300)

        T_max, J = evaluate_global_cost(sol2)
        print(f"Nivel 2: T_max = {T_max:.4f}, J_total = {J:.4e}")

        # Primero priorizamos J pequeño; entre J similares, menor T_max
        if (J < best_J) or (np.isclose(J, best_J) and T_max < best_Tmax):
            best_J = J
            best_Tmax = T_max
            best_global = sol2.copy()
            print(" -> Nueva mejor solución global")

    print("\n=== OPTIMIZACIÓN COMPLETADA ===")
    print(f"Mejor T_max = {best_Tmax:.4f}, mejor J_total = {best_J:.4e}")
    print("Mejor solución:")
    print(best_global)


if __name__ == "__main__":
    main()
