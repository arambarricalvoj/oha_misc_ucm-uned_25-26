import numpy as np
from robots import build_robot_A, build_robot_B
from solution import Solution
from sa import simulated_annealing
from evaluate import evaluate
from reachability import is_reachable_world

if __name__ == "__main__":
    np.random.seed(0)

    robotA = build_robot_A()
    robotB = build_robot_B()

    p_min = np.array([-0.2, -0.8, 0.0])
    p_max = np.array([1.5, 0.8, 1.2])

    x0 = Solution(
        pA=[0.3, 0.0, 0.5],
        qA=[1, 0, 0, 0],
        sA=0.5,
        sB=0.5,
        sRail=0.5
    )

    best, fbest, stats = simulated_annealing(
        x0=x0,
        evaluate=evaluate,
        robotA=robotA,
        robotB=robotB,
        sigma_p=0.02,
        sigma_q=0.05,
        sigma_s=0.05,
        p_min=p_min,
        p_max=p_max,
        T0=1.0,
        alpha=0.95,
        L=50,
        K=50
    )

    print("\n=== RESULTADOS ===")
    print("Mejor coste:", fbest)
    print("pA:", best.pA)
    print("pB:", best.pB)
    print("qA:", best.qA)
    print("qB:", best.qB)
    print("sA, sB, sRail:", best.sA, best.sB, best.sRail)

    print("\n--- Métricas ---")
    print("Generados:", stats["generated"])
    print("Aceptados:", stats["accepted"])
    print("Factibles:", stats["feasible"])
    print("Tasa aceptación:", stats["accepted"] / stats["generated"])
    print("Tasa factibilidad:", stats["feasible"] / stats["generated"])

    print("\n--- Verificación final ---")
    print("A:", is_reachable_world(robotA, best.pA))
    print("B:", is_reachable_world(robotB, best.pB))
