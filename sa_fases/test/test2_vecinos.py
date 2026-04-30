import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del workspace
H_MAX = 0.762
Z_MAX = 1.029

# Rango del raíl
RAIL_MIN = 0.0
RAIL_MAX = 0.70

# ---------------------------------------------------------
# Booleano para activar/desactivar guardado de imágenes
# ---------------------------------------------------------
SAVE_IMAGES = True   # <--- CAMBIA ESTO A False SI NO QUIERES GUARDAR IMÁGENES


# ---------------------------------------------------------
# Función para calcular intervalo del raíl
# ---------------------------------------------------------
def rail_interval_for_point(pB, rail_min=RAIL_MIN, rail_max=RAIL_MAX):
    px = pB[0]

    # Intervalo teórico
    r_min = px - 0.75 - H_MAX
    r_max = px - 0.75 + H_MAX

    # Recorte físico
    r_min = max(r_min, rail_min)
    r_max = min(r_max, rail_max)

    if r_min > r_max:
        return False, None, None

    return True, r_min, r_max


# ---------------------------------------------------------
# Tu función sample_feasible_point
# ---------------------------------------------------------
def sample_feasible_point(p_min, p_max,
                          offset=np.array([0.20, 0.0, 0.0]),
                          max_attempts=2000):

    cxA, cyA = 0.0, 0.0
    cxB, cyB = 0.75, 0.0

    x_min = max(p_min[0], cxA - H_MAX, cxB - H_MAX)
    x_max = min(p_max[0], cxA + H_MAX, cxB + H_MAX)

    y_min = max(p_min[1], cyA - H_MAX, cyB - H_MAX)
    y_max = min(p_max[1], cyA + H_MAX, cyB + H_MAX)

    z_min = max(p_min[2], 0.0)
    z_max = min(p_max[2], Z_MAX)

    for _ in range(max_attempts):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)

        pA = np.array([x, y, z])
        pB = pA + offset

        if (x - cxA)**2 + (y - cyA)**2 > H_MAX**2:
            continue

        if (pB[0] - cxB)**2 + (pB[1] - cyB)**2 > H_MAX**2:
            continue

        return pA, pB

    raise RuntimeError("No se ha encontrado ningún punto en la intersección de los cilindros.")


# ---------------------------------------------------------
# GENERAR 30 PUNTOS Y GUARDAR IMÁGENES INDIVIDUALES
# ---------------------------------------------------------
p_min = np.array([-1.0, -1.0, 0.0])
p_max = np.array([ 1.5,  1.0, 1.0])
offset = np.array([0.20, 0.0, 0.0])

points_A = []
points_B = []
intervals = []

for i in range(30):
    pA, pB = sample_feasible_point(p_min, p_max, offset)
    ok, rmin, rmax = rail_interval_for_point(pB)

    points_A.append(pA)
    points_B.append(pB)
    intervals.append((rmin, rmax))

    print(f"Punto {i}: pB={pB}, rail ∈ [{rmin:.3f}, {rmax:.3f}]")

    if SAVE_IMAGES:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Cilindro de A
        theta = np.linspace(0, 2*np.pi, 50)
        z = np.linspace(0, Z_MAX, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        xA = H_MAX * np.cos(theta_grid)
        yA = H_MAX * np.sin(theta_grid)
        ax.plot_surface(xA, yA, z_grid, alpha=0.15, color='blue')

        # Cilindro de B en rmin
        cxB1 = 0.75 + rmin
        xB1 = cxB1 + H_MAX * np.cos(theta_grid)
        yB1 = H_MAX * np.sin(theta_grid)
        ax.plot_surface(xB1, yB1, z_grid, alpha=0.10, color='red')

        # Cilindro de B en rmax
        cxB2 = 0.75 + rmax
        xB2 = cxB2 + H_MAX * np.cos(theta_grid)
        yB2 = H_MAX * np.sin(theta_grid)
        ax.plot_surface(xB2, yB2, z_grid, alpha=0.10, color='green')

        # Línea del raíl
        rail_x = np.linspace(0.75 + RAIL_MIN, 0.75 + RAIL_MAX, 50)
        rail_y = np.zeros_like(rail_x)
        rail_z = np.zeros_like(rail_x)
        ax.plot(rail_x, rail_y, rail_z, 'k-', linewidth=3)

        # Puntos pA y pB
        ax.scatter(pA[0], pA[1], pA[2], color='blue', s=60)
        ax.scatter(pB[0], pB[1], pB[2], color='red', s=60)

        # Offset
        ax.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], 'k--')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Punto {i} — rail ∈ [{rmin:.3f}, {rmax:.3f}]")

        plt.savefig(f"punto_{i:02d}.png")
        plt.close()


# ---------------------------------------------------------
# VISTA FINAL 3D CON TODOS LOS PUNTOS + CENTROS DE CILINDROS
# ---------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Cilindro de A
theta = np.linspace(0, 2*np.pi, 50)
z = np.linspace(0, Z_MAX, 20)
theta_grid, z_grid = np.meshgrid(theta, z)
xA = H_MAX * np.cos(theta_grid)
yA = H_MAX * np.sin(theta_grid)
ax.plot_surface(xA, yA, z_grid, alpha=0.15, color='blue')

# Cilindro de B en posición central del raíl
rail_mid = 0.5 * (RAIL_MIN + RAIL_MAX)
cxB_mid = 0.75 #+ rail_mid
xB = cxB_mid + H_MAX * np.cos(theta_grid)
yB = H_MAX * np.sin(theta_grid)
ax.plot_surface(xB, yB, z_grid, alpha=0.15, color='red')

# Puntos pA y pB
points_A = np.array(points_A)
points_B = np.array(points_B)

ax.scatter(points_A[:,0], points_A[:,1], points_A[:,2], color='blue', s=40, label='pA')
ax.scatter(points_B[:,0], points_B[:,1], points_B[:,2], color='red', s=40, label='pB')

# Líneas offset
for pA, pB in zip(points_A, points_B):
    ax.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], 'k--', alpha=0.4)

# Línea del raíl
rail_x = np.linspace(0.75 + RAIL_MIN, 0.75 + RAIL_MAX, 50)
rail_y = np.zeros_like(rail_x)
rail_z = np.zeros_like(rail_x)
ax.plot(rail_x, rail_y, rail_z, 'k-', linewidth=3, label='Raíl')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Vista final — todos los puntos")
ax.legend()

plt.show()
