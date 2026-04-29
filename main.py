import numpy as np
from robots import build_robot_A, build_robot_B
from reachability import is_reachable_world

# Construir robots
robotA = build_robot_A()
robotB = build_robot_B()

print("=== ROBOT A ===")
print(robotA)

print("\n=== ROBOT B ===")
print(robotB)

# Configuraciones articulares de prueba
qA = np.zeros(robotA.n)
qB = np.zeros(robotB.n)  # incluye el raíl: qB[0] = s

# FK de A
T_A = robotA.fkine(qA)
print("\nFK Robot A:")
print(T_A)
print("p_A_world:", T_A.t)

# FK de B
T_B = robotB.fkine(qB)
print("\nFK Robot B (con s = 0):")
print(T_B)
print("p_B_world:", T_B.t)

# Probar con un raíl desplazado
qB_shift = qB.copy()
qB_shift[0] = 0.20  # s = 0.20 m
T_B_shift = robotB.fkine(qB_shift)
print("\nFK Robot B (con s = 0.0):")
print(T_B_shift)
print("p_B_world:", T_B_shift.t)

# Prueba de alcanzabilidad
p_test = np.array([0.8, 0.1, 0.3])
print("\n=== ALCANZABILIDAD ===")
reachableA, qA_sol = is_reachable_world(robotA, p_test)
reachableB, qB_sol = is_reachable_world(robotB, p_test)

print("¿A alcanza p_test?:", reachableA)
print("qA_sol:", qA_sol)

print("¿B alcanza p_test?:", reachableB)
print("qB_sol:", qB_sol)
