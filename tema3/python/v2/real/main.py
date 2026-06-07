import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from de_timing import differential_evolution_rendezvous, DEParams
from evaluate import pA_from_x, pB_from_x, objective_components, DELTA_T_MAX


def main():
    print("\n=== EJECUCIÓN DE 30 CORRIDAS COMPLETAS (modelo robótico + IK analítica) ===\n")

    params = DEParams(pop_size=60, generations=150, F=0.75, CR=0.90)

    results = []

    for run in range(1, 30):
        print(f"--- Ejecución {run}/30 ---")
        best_x, best_comp, hist = differential_evolution_rendezvous(
            params=params,
            seed=run,
            store_history=False
        )

        pA = np.round(pA_from_x(best_x), 4)
        pB = np.round(pB_from_x(best_x), 4)

        row = {
            "run": run,
            "pA": pA,
            "pB": pB,
            "sA": round(best_x[3], 4),
            "sB": round(best_x[4], 4),
            "sRail": round(best_x[5], 4),
            "pRail": round(best_x[6], 4),
            **{
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in best_comp.items()
            }
        }
        results.append(row)

    df = pd.DataFrame(results)
    print("\nResumen de las 30 corridas:")
    print(df)

    print("\n--- Ejecución extra para visualizar convergencia ---")
    best_x, best_comp, hist = differential_evolution_rendezvous(
        params=params,
        seed=1,
        store_history=True
    )

    plt.figure(figsize=(8, 5))
    plt.plot(hist["best_J"], label="Mejor J")
    plt.plot(hist["mean_J"], label="J medio")
    plt.yscale("log")
    plt.xlabel("Generación")
    plt.ylabel("Función penalizada")
    plt.title("Convergencia de DE (modelo robótico + IK analítica)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(hist["best_Tmax"], label="Mejor Tmax")
    plt.xlabel("Generación")
    plt.ylabel("Tmax (s)")
    plt.title("Evolución del tiempo máximo")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(hist["best_delta_t"], label="|tA - tB|")
    plt.axhline(DELTA_T_MAX, linestyle="--", label="Tolerancia")
    plt.xlabel("Generación")
    plt.ylabel("Desincronización (s)")
    plt.title("Evolución de la sincronización")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
