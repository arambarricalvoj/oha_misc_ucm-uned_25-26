import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import csv

from solution import Solution
import vecinos

from nsga2_timing import nsga2_timing_level
from evaluate import compute_times


def main():

    N_RUNS = 4

    offset = np.array([0.20, 0.0, 0.0])

    p_min = np.array([-1.0, -1.0, 0.0])
    p_max = np.array([1.5, 1.0, 1.0])

    cpu_times = []
    front_sizes = []

    all_fronts = []

    print("\n=== NSGA-II ===")

    for run in range(N_RUNS):

        print(f"\n--- Ejecución {run+1}/{N_RUNS} ---")

        t0 = time.time()

        # ==================================================
        # NIVEL 1
        # ==================================================

        sol = Solution()

        sol = vecinos.generate_position_neighbor(
            sol,
            p_min,
            p_max,
            offset
        )

        # ==================================================
        # NIVEL 2
        # ==================================================

        pareto_front = nsga2_timing_level(
            sol,
            pop_size=60,
            generations=150
        )

        elapsed = time.time() - t0

        cpu_times.append(elapsed)
        front_sizes.append(len(pareto_front))

        all_fronts.append(pareto_front)

        print(
            f"Frente Pareto: {len(pareto_front)} soluciones"
        )

    print("\n=== FINALIZADO ===")

    # ======================================================
    # CSV RESUMEN
    # ======================================================

    with open(
        "resultados_nsga2.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "run",
                "cpu_time",
                "front_size"
            ]
        )

        for i in range(N_RUNS):

            writer.writerow(
                [
                    i + 1,
                    cpu_times[i],
                    front_sizes[i]
                ]
            )

    print("CSV generado.")

    # ======================================================
    # ESTADÍSTICAS
    # ======================================================

    df = pd.DataFrame(
        {
            "cpu_time": cpu_times,
            "front_size": front_sizes
        }
    )

    print("\nResumen:")

    print(df.describe())

    # ======================================================
    # ÚLTIMO FRENTE
    # ======================================================

    pareto_front = all_fronts[-1]

    times_A = []
    times_B = []

    for sol in pareto_front:

        tA, tB = compute_times(sol)

        times_A.append(tA)
        times_B.append(tB)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        times_A,
        times_B,
        c="blue",
        s=40
    )

    plt.xlabel("tA (s)")
    plt.ylabel("tB (s)")
    plt.title("Frente de Pareto final")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "pareto_front.png",
        dpi=300
    )

    plt.show()

    # ======================================================
    # TODOS LOS FRENTES
    # ======================================================

    plt.figure(figsize=(8, 6))

    for front in all_fronts:

        tA_vals = []
        tB_vals = []

        for sol in front:

            tA, tB = compute_times(sol)

            tA_vals.append(tA)
            tB_vals.append(tB)

        plt.scatter(
            tA_vals,
            tB_vals,
            alpha=0.25,
            s=10
        )

    plt.xlabel("tA (s)")
    plt.ylabel("tB (s)")
    plt.title("Frentes de Pareto (30 ejecuciones)")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "pareto_fronts_all_runs.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    main()