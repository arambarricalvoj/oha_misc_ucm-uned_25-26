"""import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV
df = pd.read_csv("resultados_individuales.csv")

# Extraer coordenadas
pA = df[["pA_x", "pA_y", "pA_z"]]
pB = df[["pB_x", "pB_y", "pB_z"]]

# ============================================================
# BOXPLOT 1: pA (x, y, z)
# ============================================================

plt.figure(figsize=(6,4))
plt.boxplot([pA["pA_x"], pA["pA_y"], pA["pA_z"]], labels=["pA_x", "pA_y", "pA_z"])
plt.title("Distribución de coordenadas de pA")
plt.ylabel("Valor")
plt.tight_layout()
plt.savefig("boxplot_pA.png", dpi=300)
plt.close()

# ============================================================
# BOXPLOT 2: pB (x, y, z)
# ============================================================

plt.figure(figsize=(6,4))
plt.boxplot([pB["pB_x"], pB["pB_y"], pB["pB_z"]], labels=["pB_x", "pB_y", "pB_z"])
plt.title("Distribución de coordenadas de pB")
plt.ylabel("Valor")
plt.tight_layout()
plt.savefig("boxplot_pB.png", dpi=300)
plt.close()

# ============================================================
# BOXPLOT 3: Comparación pA vs pB
# ============================================================

plt.figure(figsize=(8,4))
plt.boxplot([
    pA["pA_x"], pB["pB_x"],
    pA["pA_y"], pB["pB_y"],
    pA["pA_z"], pB["pB_z"]
], labels=[
    "pA_x", "pB_x",
    "pA_y", "pB_y",
    "pA_z", "pB_z"
])
plt.title("Comparación de coordenadas pA vs pB")
plt.ylabel("Valor")
plt.tight_layout()
plt.savefig("boxplot_pA_pB_comparado.png", dpi=300)
plt.close()

print("Boxplots generados.")
"""

import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV
df = pd.read_csv("resultados_individuales.csv")

# Extraer coordenadas X
pA_x = df["pA_x"]
pB_x = df["pB_x"]

# ============================================================
# BOXPLOT: pA_x vs pB_x
# ============================================================

plt.figure(figsize=(6,4))
plt.boxplot([pA_x, pB_x], labels=["pA_x", "pB_x"])
plt.title("Distribución de coordenadas X de pA y pB")
plt.ylabel("Valor en X")
plt.tight_layout()
plt.savefig("boxplot_pA_x_pB_x.png", dpi=300)
plt.close()

print("Boxplot generado: boxplot_pA_x_pB_x.png")
