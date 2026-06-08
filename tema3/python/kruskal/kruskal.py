import pandas as pd
from scipy.stats import kruskal

# Cargar los 4 CSV
df1 = pd.read_csv("resultados_individuales_f0.75-cr0.9.csv")
df2 = pd.read_csv("resultados_individuales_f0.6-cr0.7.csv")
df3 = pd.read_csv("resultados_individuales_f0.9-cr0.9.csv")
df4 = pd.read_csv("resultados_individuales_f0.4-cr0.5.csv")

# Extraer la métrica que quieres comparar
t1 = df1["Tmax"]
t2 = df2["Tmax"]
t3 = df3["Tmax"]
t4 = df4["Tmax"]

# Test de Kruskal–Wallis
stat, p = kruskal(t1, t2, t3, t4)

print("Estadístico H:", stat)
print("p-value:", p)

if p < 0.05:
    print("Conclusión: Hay diferencias significativas entre configuraciones.")
else:
    print("Conclusión: No se detectan diferencias significativas entre configuraciones.")
