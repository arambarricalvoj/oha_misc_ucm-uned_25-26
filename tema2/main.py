import numpy as np
import time
import matplotlib.pyplot as plt
import csv

from solution import Solution
import vecinos
from sa_timing import sa_timing_level
from evaluate import evaluate_global_cost


def main():
    N_RUNS = 30
    offset = np.array([0.20, 0.0, 0.0])

    p_min = np.array([-1.0, -1.0, 0.0])
    p_max = np.array([ 1.5,  1.0, 1.0])

    # === LISTAS PARA MÉTRICAS ===
    Tmax_list = []
    time_list = []
    iter_list = []
    eff_list = []
    history_list = []

    # === LISTAS PARA VECTOR DE SOLUCIÓN ===
    sA_list = []
    sB_list = []
    sRail_list = []
    pRail_list = []

    # === POSICIONES pA y pB ===
    pA_list = []
    pB_list = []

    print("\n=== EJECUCIÓN DE 30 CORRIDAS COMPLETAS ===")

    for run in range(N_RUNS):

        t0 = time.time()

        print(f"\n--- Ejecución {run+1}/{N_RUNS} ---")

        sol = Solution()
        sol = vecinos.generate_position_neighbor(sol, p_min, p_max, offset)

        sol2, J_best, best_iter, history_J = sa_timing_level(sol, max_iters=300)

        Tmax, J_total = evaluate_global_cost(sol2)

        t1 = time.time()
        elapsed = t1 - t0

        print(f"Tmax = {Tmax:.4f}, tiempo real = {elapsed:.3f}s, iter óptima = {best_iter}")

        # === GUARDAR MÉTRICAS ===
        Tmax_list.append(Tmax)
        time_list.append(elapsed)
        iter_list.append(best_iter)
        eff_list.append(Tmax / elapsed)
        history_list.append(history_J)

        # === GUARDAR VECTOR DE SOLUCIÓN ===
        sA_list.append(sol2.sA)
        sB_list.append(sol2.sB)
        sRail_list.append(sol2.sRail)
        pRail_list.append(sol2.pRail)

        # === GUARDAR POSICIONES pA y pB ===
        pA_list.append(sol2.pA.copy())
        pB_list.append(sol2.pB.copy())

    print("\n=== TODAS LAS EJECUCIONES COMPLETADAS ===")

    # ============================================================
    # CSV: RESULTADOS INDIVIDUALES
    # ============================================================

    with open("resultados_individuales.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ejecución", "Tmax", "Tiempo (s)", "Iteración óptima", "Eficiencia",
            "sA", "sB", "sRail", "pRail",
            "pA_x", "pA_y", "pA_z",
            "pB_x", "pB_y", "pB_z"
        ])

        for i in range(N_RUNS):
            writer.writerow([
                i+1,
                Tmax_list[i],
                time_list[i],
                iter_list[i],
                eff_list[i],
                sA_list[i],
                sB_list[i],
                sRail_list[i],
                pRail_list[i],
                pA_list[i][0], pA_list[i][1], pA_list[i][2],
                pB_list[i][0], pB_list[i][1], pB_list[i][2]
            ])

    print("CSV generado: resultados_individuales.csv")

    # ============================================================
    # CSV: MÉTRICAS AGREGADAS
    # ============================================================

    Tmax_arr = np.array(Tmax_list)
    time_arr = np.array(time_list)

    mean_T = np.mean(Tmax_arr)
    std_T = np.std(Tmax_arr)
    min_T = np.min(Tmax_arr)
    max_T = np.max(Tmax_arr)
    ci_low = mean_T - 1.96 * std_T / np.sqrt(N_RUNS)
    ci_high = mean_T + 1.96 * std_T / np.sqrt(N_RUNS)
    mean_time = np.mean(time_arr)

    with open("resultados_agregados.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Métrica", "Valor"])
        writer.writerow(["Media Tmax", mean_T])
        writer.writerow(["Desviación típica", std_T])
        writer.writerow(["Mejor valor (mínimo)", min_T])
        writer.writerow(["Peor valor (máximo)", max_T])
        writer.writerow(["IC 95% (low)", ci_low])
        writer.writerow(["IC 95% (high)", ci_high])
        writer.writerow(["Tiempo medio (s)", mean_time])

    print("CSV generado: resultados_agregados.csv")

    # ============================================================
    # FIGURA 1: DIAGRAMA DE DISPERSIÓN PROPUESTO
    # ============================================================

    plt.figure(figsize=(8,6))

    # Línea de tendencia
    z = np.polyfit(time_list, Tmax_list, 1)
    p = np.poly1d(z)
    plt.plot(time_list, p(time_list), "k--", linewidth=1.5, label="Tendencia")

    # Mejor y peor solución
    idx_best = np.argmin(Tmax_list)
    idx_worst = np.argmax(Tmax_list)

    # Puntos normales
    scatter = plt.scatter(time_list, Tmax_list, c=eff_list, cmap='viridis', s=80)

    # Etiquetas de ejecución
    for i in range(N_RUNS):
        plt.text(time_list[i] + 0.002, Tmax_list[i] + 0.002, str(i+1), fontsize=8)

    # Mejor solución → borde negro + verde
    plt.scatter(time_list[idx_best], Tmax_list[idx_best],
                edgecolors='black', facecolors='green', s=200, label="Mejor solución")

    # Peor solución → borde negro + rojo
    plt.scatter(time_list[idx_worst], Tmax_list[idx_worst],
                edgecolors='black', facecolors='red', s=200, label="Peor solución")

    plt.xlabel("Tiempo real (s)")
    plt.ylabel("Tmax")
    plt.title("Diagrama de dispersión propuesto para la eficiencia")
    plt.colorbar(scatter, label="Eficiencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig("scatter_plot_propuesto.png", dpi=300)
    plt.close()

    # ============================================================
    # FIGURA 2: BOXPLOT
    # ============================================================

    plt.figure(figsize=(6,4))
    plt.boxplot(Tmax_list)
    plt.ylabel("Tmax")
    plt.title("Distribución de Tmax (30 ejecuciones)")
    plt.savefig("boxplot_Tmax.png", dpi=300)
    plt.close()

    # ============================================================
    # FIGURA 3A: CURVAS DE CONVERGENCIA INDIVIDUALES
    # ============================================================

    plt.figure(figsize=(8,5))

    for h in history_list:
        plt.plot(h, alpha=0.35, linewidth=1.2, color='steelblue')

    plt.xlabel("Iteración")
    plt.ylabel("J")
    plt.title("Curvas de convergencia del Temple Simulado")
    plt.xlim(0, 150)
    plt.tight_layout()
    plt.savefig("convergencia_individuales.png", dpi=300)
    plt.close()

    # ============================================================
    # FIGURA 3B: CURVA MEDIA DE CONVERGENCIA
    # ============================================================

    plt.figure(figsize=(8,5))

    max_len = max(len(h) for h in history_list)
    histories = np.array([np.pad(h, (0, max_len - len(h)), 'edge') for h in history_list])

    mean_curve = histories.mean(axis=0)
    std_curve = histories.std(axis=0)

    plt.fill_between(
        np.arange(max_len),
        mean_curve - std_curve,
        mean_curve + std_curve,
        color='gray',
        alpha=0.25,
        label='±1 desviación típica'
    )

    plt.plot(mean_curve, color='black', linewidth=2.5, label='Media')

    plt.xlabel("Iteración")
    plt.ylabel("J")
    plt.title("Curva media de convergencia del Temple Simulado")
    plt.xlim(0, 150)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergencia_media.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
