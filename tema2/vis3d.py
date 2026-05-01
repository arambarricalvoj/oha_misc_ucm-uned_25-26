import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del workspace
H_MAX = 0.762
Z_MAX = 1.029
RAIL_MIN = 0.0
RAIL_MAX = 0.70

# Cargar CSV
df = pd.read_csv("resultados_individuales.csv")

# Extraer puntos pA y pB
pA = df[["pA_x", "pA_y", "pA_z"]].values
pB = df[["pB_x", "pB_y", "pB_z"]].values

# Crear figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# ---------------------------------------------------------
# Cilindro de A
# ---------------------------------------------------------
theta = np.linspace(0, 2*np.pi, 50)
z = np.linspace(0, Z_MAX, 20)
theta_grid, z_grid = np.meshgrid(theta, z)

xA = H_MAX * np.cos(theta_grid)
yA = H_MAX * np.sin(theta_grid)
ax.plot_surface(xA, yA, z_grid, alpha=0.15, color='blue')

# ---------------------------------------------------------
# Cilindro de B (posición central del raíl)
# ---------------------------------------------------------
cxB = 0.75
xB = cxB + H_MAX * np.cos(theta_grid)
yB = H_MAX * np.sin(theta_grid)
ax.plot_surface(xB, yB, z_grid, alpha=0.15, color='red')

# ---------------------------------------------------------
# Puntos pA y pB
# ---------------------------------------------------------
ax.scatter(pA[:,0], pA[:,1], pA[:,2], color='blue', s=40, label='pA')
ax.scatter(pB[:,0], pB[:,1], pB[:,2], color='red', s=40, label='pB')

# Líneas offset
for a, b in zip(pA, pB):
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'k--', alpha=0.4)

# ---------------------------------------------------------
# Línea del raíl
# ---------------------------------------------------------
rail_x = np.linspace(0.75 + RAIL_MIN, 0.75 + RAIL_MAX, 50)
rail_y = np.zeros_like(rail_x)
rail_z = np.zeros_like(rail_x)
ax.plot(rail_x, rail_y, rail_z, 'k-', linewidth=3, label='Raíl')

# ---------------------------------------------------------
# Ajustes finales
# ---------------------------------------------------------
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Visualización 3D de pA y pB obtenidos por el algoritmo")
ax.legend()

plt.tight_layout()
plt.savefig("visualizacion_pA_pB.png", dpi=300)
plt.show()
