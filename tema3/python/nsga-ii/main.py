import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import csv

from solution import Solution
import vecinos

from nsga2_timing import nsga2_timing_level
from evaluate import compute_times


# ============================================================
# MÉTRICAS MULTIOBJETIVO
# ============================================================

def hypervolume(front, ref):
    hv = 0.0
    for f in front:
        hv += max(0, (ref[0] - f[0]) * (ref[1] - f[1]))
    return hv

def diversity(front):
    front = sorted(front, key=lambda x: x[0])
    distances = [np.linalg.norm(front[i] - front[i-1]) for i in range(1, len(front))]
    return np.std(distances)


# ============================================================
# MAIN
# ============================================================

def main():

    N_RUNS = 4

    offset = np.array([0.20, 0.0, 0.0])

    p_min = np.array([-1.0, -1.0, 0.0])
    p_max = np.array([1.5, 1.0, 1.0])

    cpu_times = []
    front_sizes = []
    all_fronts = []

    hv_list = []
    div_list = []

    print("\n=== NSGA-II ===")

    for run in range(N_RUNS):

        print(f"\n--- Ejecución {run+1}/{N_RUNS} ---")

        t0 = time.time()

        # ==================================================
        # NIVEL 1: generar posición
        # ==================================================

        sol = Solution()

        sol = vecinos.generate_position_neighbor(
            sol,
            p_min,
            p_max,
            offset
        )

        # ==================================================
        # NIVEL 2: NSGA-II
        # ==================================================

        pareto_front, log = nsga2_timing_level(
            sol,
            pop_size=20,
            generations=150
        )

        # ============================================================
        # EXPORTAR FRENTE DE PARETO COMPLETO
        # ============================================================

        with open(f"pareto_run_{run+1}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sA","sB","sRail","pRail",
                "pA_x","pA_y","pA_z",
                "pB_x","pB_y","pB_z",
                "qA1","qA2","qA3","qA4","qA5","qA6",
                "qB1","qB2","qB3","qB4","qB5","qB6",
                "Tmax","DeltaT"
            ])

            for sol in pareto_front:
                tA, tB = compute_times(sol)
                Tmax = max(tA, tB)
                DeltaT = abs(tA - tB)

                writer.writerow([
                    sol.sA, sol.sB, sol.sRail, sol.pRail,
                    sol.pA[0], sol.pA[1], sol.pA[2],
                    sol.pB[0], sol.pB[1], sol.pB[2],
                    sol.qA[0], sol.qA[1], sol.qA[2], sol.qA[3], sol.qA[4], sol.qA[5],
                    sol.qB[0], sol.qB[1], sol.qB[2], sol.qB[3], sol.qB[4], sol.qB[5],
                    Tmax, DeltaT
                ])

        # ============================================================
        # EXPORTAR LOG COMPLETO
        # ============================================================

        with open(f"log_run_{run+1}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gen", "sA", "sB", "sRail", "pRail", "Tmax", "DeltaT"])

            for gen_idx, gen_data in enumerate(log):
                for row in gen_data:
                    writer.writerow([gen_idx] + row)

        # ============================================================
        # MÉTRICAS POR GENERACIÓN
        # ============================================================

        metrics = []

        for gen_idx, gen_data in enumerate(log):

            objs = np.array([
                [row[4], row[5]]   # Tmax, DeltaT
                for row in gen_data
            ])

            ref = np.max(objs, axis=0) * 1.2

            hv = hypervolume(objs, ref)
            div = diversity(objs)

            metrics.append([gen_idx, hv, div])

        with open(f"metrics_run_{run+1}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gen", "HV", "diversity"])
            writer.writerows(metrics)

        # ============================================================
        # MÉTRICAS FINALES DEL FRENTE
        # ============================================================

        objs = np.array([
            [max(compute_times(sol)), abs(compute_times(sol)[0] - compute_times(sol)[1])]
            for sol in pareto_front
        ])

        ref = np.max(objs, axis=0) * 1.2

        hv = hypervolume(objs, ref)
        div = diversity(objs)

        hv_list.append(hv)
        div_list.append(div)

        print(f"Run {run+1}: HV={hv}, diversity={div}")

        elapsed = time.time() - t0

        cpu_times.append(elapsed)
        front_sizes.append(len(pareto_front))
        all_fronts.append(pareto_front)

        print(f"Frente Pareto: {len(pareto_front)} soluciones")

    print("\n=== FINALIZADO ===")

    # ======================================================
    # IDENTIFICAR LA MEJOR EJECUCIÓN
    # ======================================================

    best_run = np.argmax(hv_list)
    print(f"\n>>> Mejor ejecución según Hipervolumen: RUN {best_run+1}\n")

    # ======================================================
    # CSV RESUMEN
    # ======================================================

    with open("resultados_nsga2.csv", "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["run", "cpu_time", "front_size", "HV", "diversity"])

        for i in range(N_RUNS):
            writer.writerow([i + 1, cpu_times[i], front_sizes[i], hv_list[i], div_list[i]])

    print("CSV generado.")

    # ======================================================
    # CSV: MÉTRICAS AGREGADAS
    # ======================================================

    hv_arr = np.array(hv_list)
    div_arr = np.array(div_list)
    time_arr = np.array(cpu_times)

    def stats(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        ci_low = mean - 1.96 * std / np.sqrt(N_RUNS)
        ci_high = mean + 1.96 * std / np.sqrt(N_RUNS)
        return mean, std, np.min(arr), np.max(arr), ci_low, ci_high

    hv_stats = stats(hv_arr)
    div_stats = stats(div_arr)

    with open("resultados_agregados.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Métrica", "Media", "Std", "Min", "Max", "IC_low", "IC_high"])
        writer.writerow(["HV"] + list(hv_stats))
        writer.writerow(["diversity"] + list(div_stats))
        writer.writerow(["Tiempo medio (s)", np.mean(time_arr)])

    print("CSV generado: resultados_agregados.csv")

    # ======================================================
    # CURVAS DE MÉTRICAS (INDIVIDUALES + MEDIA)
    # ======================================================

    metric_names = ["HV", "diversity"]

    for metric in metric_names:

        histories = []
        max_len = 0

        for run in range(N_RUNS):
            df = pd.read_csv(f"metrics_run_{run+1}.csv")
            h = df[metric].values
            histories.append(h)
            max_len = max(max_len, len(h))

        # Curvas individuales
        plt.figure(figsize=(8,5))
        for h in histories:
            plt.plot(h, alpha=0.35, linewidth=1.2)
        plt.xlabel("Generación")
        plt.ylabel(metric)
        plt.title(f"Curvas individuales de {metric} (NSGA-II)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{metric}_individuales.png", dpi=300)
        plt.close()

        # Curva media
        padded = np.array([np.pad(h, (0, max_len - len(h)), 'edge') for h in histories])
        mean_curve = padded.mean(axis=0)
        std_curve = padded.std(axis=0)

        plt.figure(figsize=(8,5))
        plt.fill_between(
            np.arange(max_len),
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.25,
            label="±1 desviación típica"
        )
        plt.plot(mean_curve, color="black", linewidth=2.5, label="Media")
        plt.xlabel("Generación")
        plt.ylabel(metric)
        plt.title(f"Curva media de {metric} (NSGA-II)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{metric}_media.png", dpi=300)
        plt.close()

    # ======================================================
    # FRENTE FINAL (MEJOR EJECUCIÓN)
    # ======================================================

    pareto_front = all_fronts[best_run]

    times_A = []
    times_B = []

    for sol in pareto_front:
        tA, tB = compute_times(sol)
        times_A.append(tA)
        times_B.append(tB)

    plt.figure(figsize=(8, 6))
    plt.scatter(times_A, times_B, c="blue", s=80, edgecolor="black")
    plt.xlabel("tA (s)")
    plt.ylabel("tB (s)")
    plt.title(f"Frente de Pareto final (mejor ejecución: run {best_run+1})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=300)
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

        plt.scatter(tA_vals, tB_vals, alpha=0.25, s=10)

    plt.xlabel("tA (s)")
    plt.ylabel("tB (s)")
    plt.title("Frentes de Pareto (todas las ejecuciones)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pareto_fronts_all_runs.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
