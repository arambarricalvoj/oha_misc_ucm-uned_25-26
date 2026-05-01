import numpy as np
from spatialmath import SE3
from robots import build_robot_B, ik_B

# ============================
# CONFIGURACIÓN
# ============================

robotB = build_robot_B()

# Orientación fija que tú definiste:
#   RB = [0, +90°, 180°]
# RB = np.array([0, np.pi/2, 0])
RB = np.array([np.pi, np.pi/2, 0])

# Semilla de IK
q_seed = np.zeros(6)

# Raíl fijo para el test
pRail = 0.30

# Rango de puntos a testear
xs = np.linspace(0.50, 1.00, 6)   # delante del robot B
ys = np.linspace(-0.40, 0.40, 5)  # lateral
zs = np.linspace(0.30, 1.00, 6)   # altura

failed_points = []
success_points = []

print("\n=== BARRIDO DE IK PARA ROBOT B ===")

for x in xs:
    for y in ys:
        for z in zs:
            pB = np.array([x, y, z])

            qB, okB = ik_B(pB, RB, q_seed, pRail)

            if okB:
                success_points.append(pB)
            else:
                failed_points.append(pB)

# ============================
# RESULTADOS
# ============================

print("\nTotal puntos probados:", len(xs)*len(ys)*len(zs))
print("IK OK:", len(success_points))
print("IK FAIL:", len(failed_points))

print("\n=== PUNTOS FALLIDOS ===")
for p in failed_points:
    print("  ", p)

print("\n=== PUNTOS OK ===")
for p in success_points[:10]:
    print("  ", p)
if len(success_points) > 10:
    print("  ...")
