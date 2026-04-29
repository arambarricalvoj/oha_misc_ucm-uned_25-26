import numpy as np
from robots import build_robot_A, build_robot_B
from reachability import is_reachable_world, rail_interval, reachable_with_rail

# Construir robots
robotA = build_robot_A()
robotB = build_robot_B()

# Rango de puntos a testear
X = np.linspace(-0.2, 1.5, 40)
Y = np.linspace(-0.8, 0.8, 40)
Z = np.linspace(0.0, 1.2, 20)

total = 0
A_reach = 0
B_reach = 0
B_rail_needed = 0
B_unreachable = 0

print("Iniciando test masivo de alcanzabilidad...")

for x in X:
    for y in Y:
        for z in Z:
            p = np.array([x, y, z])
            total += 1

            # Robot A
            A_ok, _ = is_reachable_world(robotA, p)
            if A_ok:
                A_reach += 1

            # Robot B
            B_ok, solB = is_reachable_world(robotB, p)
            if B_ok:
                B_reach += 1
                s, q = solB
                if s > 0.001:
                    B_rail_needed += 1
            else:
                B_unreachable += 1

print("\n=== RESULTADOS ===")
print(f"Puntos totales evaluados: {total}")
print(f"Robot A alcanza: {A_reach}")
print(f"Robot B alcanza: {B_reach}")
print(f"Robot B necesita mover el raíl en: {B_rail_needed} puntos")
print(f"Puntos NO alcanzables por B: {B_unreachable}")
