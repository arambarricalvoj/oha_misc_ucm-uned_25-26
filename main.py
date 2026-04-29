import numpy as np
from robots import build_robot_A, build_robot_B
from reachability import is_reachable_world, rail_interval

# Construir robots
robotA = build_robot_A()
robotB = build_robot_B()

print("=== ROBOT A ===")
print(robotA)

print("\n=== ROBOT B ===")
print(robotB)

# Configuraciones articulares de prueba
qA = np.zeros(robotA.n)
qB = np.zeros(robotB.n)

# FK de A
T_A = robotA.fkine(qA)
print("\nFK Robot A:")
print(T_A)
print("p_A_world:", T_A.t)

# FK de B
T_B = robotB.fkine(qB)
print("\nFK Robot B (base fija + raíl):")
print(T_B)
print("p_B_world:", T_B.t)

# Prueba de alcanzabilidad
p_test = np.array([0.8, 0.1, 0.3])
print("\n=== ALCANZABILIDAD ===")

reachableA, solA = is_reachable_world(robotA, p_test)
reachableB, solB = is_reachable_world(robotB, p_test)

print("¿A alcanza p_test?:", reachableA)
print("Solución A:", solA)

print("¿B alcanza p_test?:", reachableB)
print("Solución B (s, q):", solB)

# Intervalo de alcanzabilidad del raíl
reachable_interval, s_min, s_max = rail_interval(robotB, p_test)

print("\n=== INTERVALO DE ALCANZABILIDAD DEL RAÍL (Robot B) ===")
print("¿Existe intervalo?:", reachable_interval)
print("s_min:", s_min)
print("s_max:", s_max)

if reachable_interval:
    print(f"\nEl punto es alcanzable para s en [{s_min:.3f}, {s_max:.3f}] metros.")
    print(f"El raíl debe moverse al menos {s_min:.3f} m desde su posición actual (s=0).")
else:
    print("\nEl punto NO es alcanzable para ningún valor del raíl.")
