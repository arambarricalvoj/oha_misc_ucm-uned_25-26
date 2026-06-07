import numpy as np

# Parámetros del workspace
H_MAX = 0.762
Z_MAX = 1.029

# Rango físico del raíl
RAIL_MIN = 0.0
RAIL_MAX = 0.70

# Manipulabilidad mínima aceptable
MANIP_MIN = 0.08


# ============================================================
#  GENERACIÓN DE PUNTOS GEOMÉTRICAMENTE FACTIBLES
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

        # Dentro del cilindro de A
        if (x - cxA)**2 + (y - cyA)**2 > H_MAX**2:
            continue

        # Dentro del cilindro de B
        if (pB[0] - cxB)**2 + (pB[1] - cyB)**2 > H_MAX**2:
            continue

        return pA, pB

    raise RuntimeError("No se ha encontrado ningún punto en la intersección de los cilindros.")


# ============================================================
#  INTERVALO DEL RAÍL PARA UN PUNTO pB
# ============================================================

def rail_interval_for_point(pB, rail_min=RAIL_MIN, rail_max=RAIL_MAX):
    """
    Devuelve el intervalo [r_min, r_max] de posiciones del raíl
    que permiten alcanzar el punto pB.
    """
    px = pB[0]

    r_min = px - 0.75 - H_MAX
    r_max = px - 0.75 + H_MAX

    r_min = max(r_min, rail_min)
    r_max = min(r_max, rail_max)

    if r_min > r_max:
        return False, None, None

    return True, r_min, r_max


# ============================================================
#  NIVEL 1: GENERACIÓN DE PUNTOS + FILTRO IK + MANIPULABILIDAD
# ============================================================

def generate_position_neighbor(sol, p_min, p_max, offset):
    """
    Nivel 1: genera un nuevo punto pA/pB válido, factible por IK
    y con buena manipulabilidad.
    """
    from solution import Solution
    from robots import ik_A, ik_B, manipulability, build_robot_A, build_robot_B

    if sol is None:
        sol = Solution()

    MAX_TRIES = 200

    for _ in range(MAX_TRIES):

        # 1. Punto geométricamente válido
        pA, pB = sample_feasible_point(p_min, p_max, offset)

        # 2. Intervalo del raíl
        ok, rmin, rmax = rail_interval_for_point(pB)
        if not ok:
            continue

        # Elegimos un pRail intermedio para probar IK
        pRail_test = 0.5 * (rmin + rmax)

        # 3. IK A
        qA, okA = ik_A(pA, sol.RA, np.zeros(6))
        if not okA:
            continue

        # 4. IK B
        qB, okB = ik_B(pB, sol.RB, np.zeros(6), pRail_test)
        if not okB:
            continue

        # 5. Manipulabilidad
        mA = manipulability(build_robot_A(), qA)
        mB = manipulability(build_robot_B(), qB)

        if mA < MANIP_MIN or mB < MANIP_MIN:
            continue

        # 6. Aceptar punto
        new = sol.copy()
        new.pA = pA
        new.pB = pB
        new.rail_min = rmin
        new.rail_max = rmax
        new.pRail = pRail_test
        new.qA = qA
        new.qB = qB
        return new

    # Si no se encuentra nada, devolvemos copia sin cambios
    return sol.copy()


# ============================================================
#  NIVEL 2: MUTACIÓN DE VELOCIDADES Y RAÍL
# ============================================================

def generate_timing_neighbor(sol, sigma_s=0.05, sigma_prail=0.05):
    """
    Nivel 2: muta sA, sB, sRail y pRail (respetando [rail_min, rail_max]).
    """
    new = sol.copy()

    new.sA = np.clip(new.sA + np.random.normal(0, sigma_s), 0.0, 1.0)
    new.sB = np.clip(new.sB + np.random.normal(0, sigma_s), 0.0, 1.0)
    new.sRail = np.clip(new.sRail + np.random.normal(0, sigma_s), 0.0, 1.0)

    new.pRail += np.random.normal(0, sigma_prail)
    new.pRail = np.clip(new.pRail, new.rail_min, new.rail_max)

    return new
