import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from robots import build_robot_A, build_robot_B
from reachability import reachable_with_rail, ik_with_rail

# Parámetros del workspace
H_MAX = 0.762
Z_MAX = 1.029

# Construir robots
robotA = build_robot_A()
robotB = build_robot_B()

# Crear figura 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# ---------------------------------------------------------
# 1. Workspace del Robot A (cilindro fijo)
# ---------------------------------------------------------
theta = np.linspace(0, 2*np.pi, 40)
z = np.linspace(0, Z_MAX, 20)
theta_grid, z_grid = np.meshgrid(theta, z)

xA = H_MAX * np.cos(theta_grid)
yA = H_MAX * np.sin(theta_grid)
zA = z_grid

ax.plot_surface(xA, yA, zA, alpha=0.15, color='green')

# ---------------------------------------------------------
# 2. Workspace del Robot B (túnel del raíl)
# ---------------------------------------------------------
rail_positions = np.linspace(0, 0.70, 6)

for s in rail_positions:
    base_x = robotB.base.t[0] + s
    xB = base_x + H_MAX * np.cos(theta_grid)
    yB = H_MAX * np.sin(theta_grid)
    zB = z_grid
    ax.plot_surface(xB, yB, zB, alpha=0.10, color='blue')

# ---------------------------------------------------------
# 3. Generar 40 puntos de test
# ---------------------------------------------------------
np.random.seed(0)
points = np.random.uniform(
    low=[-0.2, -0.8, 0.0],
    high=[1.5, 0.8, 1.2],
    size=(40, 3)
)

print("\n=== RESULTADOS IK PARA PUNTOS ✓ ===")

for p in points:
    # Alcanzabilidad geométrica
    A_ok = reachable_with_rail(robotA, p, s=0)
    B_ok_s0 = reachable_with_rail(robotB, p, s=0)
    B_ok_any = any(reachable_with_rail(robotB, p, s) for s in np.linspace(0, 0.70, 40))

    if A_ok and B_ok_s0:
        # Dibujar ✓
        ax.text(p[0], p[1], p[2], "✓", color='black', fontsize=14, ha='center')

        # ---- IK CHECK ----
        solA = ik_with_rail(robotA, p, s=0)
        solB = ik_with_rail(robotB, p, s=0)

        print(f"\nPunto {p} → ✓ (intersección geométrica)")
        print(f"  IK A: {'OK' if solA.success else 'NO'}")
        print(f"  IK B: {'OK' if solB.success else 'NO'}")

    elif A_ok:
        ax.text(p[0], p[1], p[2], "A", color='green', fontsize=14, ha='center')

    elif B_ok_any:
        ax.text(p[0], p[1], p[2], "B", color='blue', fontsize=14, ha='center')

    else:
        ax.text(p[0], p[1], p[2], "X", color='red', fontsize=14, ha='center')

# ---------------------------------------------------------
# Ajustes de la gráfica
# ---------------------------------------------------------
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Workspaces de Robot A (verde) y Robot B (azul) + puntos de test")

ax.set_xlim(-0.2, 1.6)
ax.set_ylim(-0.8, 0.8)
ax.set_zlim(0, 1.2)

plt.show()
