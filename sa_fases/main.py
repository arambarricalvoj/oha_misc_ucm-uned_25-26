import numpy as np
import vecinos
from solution import Solution


def main():

    # Bounding box para pA
    p_min = np.array([-1.0, -1.0, 0.0])
    p_max = np.array([ 1.5,  1.0, 1.0])
    offset = np.array([0.20, 0.0, 0.0])

    print("\n=== TEST NIVEL 1: Generación de puntos factibles ===")

    for i in range(5):
        print(f"\n--- Generando punto {i} ---")

        sol = Solution()
        sol = vecinos.generate_position_neighbor(sol, p_min, p_max, offset)

        print(sol)


if __name__ == "__main__":
    main()
