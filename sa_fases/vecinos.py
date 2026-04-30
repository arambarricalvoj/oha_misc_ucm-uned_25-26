import numpy as np

# Parámetros del workspace
H_MAX = 0.762
Z_MAX = 1.029

# Rango físico del raíl
RAIL_MIN = 0.0
RAIL_MAX = 0.70


# ============================================================
# NIVEL 1 — Generación de puntos factibles pA, pB
# ============================================================

def sample_feasible_point(p_min, p_max,
                          offset=np.array([0.20, 0.0, 0.0]),
                          max_attempts=2000):
    """
    Genera un punto pA tal que:
    - pA está dentro del cilindro de A
    - pB = pA + offset está dentro del cilindro de B
    """

    cxA, cyA = 0.0, 0.0
    cxB, cyB = 0.75, 0.0

    # Bounding box aproximado de la intersección
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

        # Cilindro de A
        if (x - cxA)**2 + (y - cyA)**2 > H_MAX**2:
            continue

        # Cilindro de B aplicado a pB
        if (pB[0] - cxB)**2 + (pB[1] - cyB)**2 > H_MAX**2:
            continue

        return pA, pB

    raise RuntimeError("No se ha encontrado ningún punto en la intersección de los cilindros.")



# ============================================================
# Cálculo del intervalo del raíl para un punto pB
# ============================================================

def rail_interval_for_point(pB, rail_min=RAIL_MIN, rail_max=RAIL_MAX):
    """
    Devuelve el intervalo [r_min, r_max] de posiciones del raíl
    que permiten alcanzar el punto pB.
    """

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



# ============================================================
# NIVEL 1 — Vecino de posición (nuevo pA/pB)
# ============================================================

def generate_position_neighbor(sol, p_min, p_max, offset):
    """
    Genera un nuevo punto pA/pB válido.
    Se usa en el nivel 1 del SA.
    """

    pA, pB = sample_feasible_point(p_min, p_max, offset)
    ok, rmin, rmax = rail_interval_for_point(pB)

    if not ok:
        # No debería ocurrir, pero por seguridad
        return sol.copy()

    new = sol.copy()
    new.pA = pA
    new.pB = pB
    new.rail_min = rmin
    new.rail_max = rmax

    # Elegimos una posición del raíl válida (por ejemplo, el centro)
    new.pRail = 0.5 * (rmin + rmax)

    return new



# ============================================================
# NIVEL 2 — Vecino de tiempos (sA, sB, sRail, pRail)
# ============================================================

def generate_timing_neighbor(sol, sigma_s=0.05, sigma_prail=0.05):
    """
    Genera un vecino mutando:
    - sA, sB (velocidades)
    - sRail (velocidad del raíl)
    - pRail (posición del raíl, respetando [rail_min, rail_max])
    """

    new = sol.copy()

    # Mutar velocidades
    new.sA = np.clip(new.sA + np.random.normal(0, sigma_s), 0, 1)
    new.sB = np.clip(new.sB + np.random.normal(0, sigma_s), 0, 1)
    new.sRail = np.clip(new.sRail + np.random.normal(0, sigma_s), 0, 1)

    # Mutar posición del raíl dentro del intervalo permitido
    new.pRail += np.random.normal(0, sigma_prail)
    new.pRail = np.clip(new.pRail, new.rail_min, new.rail_max)

    return new
