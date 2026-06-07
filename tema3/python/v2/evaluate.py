import numpy as np

# Geometría simplificada
R_A = 0.762
R_B = 0.762
HEIGHT = 1.029
D_EF = 0.20
BASE_B_X = 0.75
RAIL_MIN, RAIL_MAX = 0.0, 0.70

# Velocidades nominales
V_ARM_A = 0.60
V_ARM_B = 0.55
V_RAIL = 0.35

# Tolerancia de sincronización
DELTA_T_MAX = 0.05

# Penalizaciones
LAMBDA_SYNC = 50.0
LAMBDA_FEAS = 1000.0

# Variables: pAx, pAy, pAz, sA, sB, sRail, pRail
LOWER = np.array([-0.20, -0.60, 0.05, 0.05, 0.05, 0.05, RAIL_MIN])
UPPER = np.array([ 0.70,  0.60, 0.95, 1.00, 1.00, 1.00, RAIL_MAX])
DIM = len(LOWER)

P_A_START = np.array([0.0, -0.35, 0.30])
P_B_START_WORLD = np.array([BASE_B_X, 0.35, 0.30])


def pA_from_x(x):
    return x[:3]


def pB_from_x(x):
    return pA_from_x(x) + np.array([D_EF, 0.0, 0.0])


def rail_from_x(x):
    return x[6]


def speeds_from_x(x):
    return x[3], x[4], x[5]


def workspace_violation_A(pA):
    radial = max(0.0, np.sqrt(pA[0]**2 + pA[1]**2) - R_A)
    z_low = max(0.0, -pA[2])
    z_high = max(0.0, pA[2] - HEIGHT)
    return radial**2 + z_low**2 + z_high**2


def workspace_violation_B(pB, pRail):
    base_x = BASE_B_X + pRail
    radial = max(0.0, np.sqrt((pB[0] - base_x)**2 + pB[1]**2) - R_B)
    z_low = max(0.0, -pB[2])
    z_high = max(0.0, pB[2] - HEIGHT)
    rail_low = max(0.0, RAIL_MIN - pRail)
    rail_high = max(0.0, pRail - RAIL_MAX)
    return radial**2 + z_low**2 + z_high**2 + rail_low**2 + rail_high**2


def feasibility_violation(x):
    pA = pA_from_x(x)
    pB = pB_from_x(x)
    return workspace_violation_A(pA) + workspace_violation_B(pB, rail_from_x(x))


def is_feasible(x, tol=1e-10):
    return feasibility_violation(x) <= tol


def clip_bounds(x):
    return np.minimum(np.maximum(x, LOWER), UPPER)


def repair_solution(x):
    x = clip_bounds(x.copy())
    # Aproximación: situar el raíl para acercar la base de B al punto pB
    pB = pB_from_x(x)
    x[6] = np.clip(pB[0] - BASE_B_X, RAIL_MIN, RAIL_MAX)
    return clip_bounds(x)


def times(x):
    x = repair_solution(x)
    pA = pA_from_x(x)
    pB = pB_from_x(x)
    sA, sB, sRail = speeds_from_x(x)
    pRail = rail_from_x(x)

    sA = max(sA, 1e-6)
    sB = max(sB, 1e-6)
    sRail = max(sRail, 1e-6)

    length_A = np.linalg.norm(pA - P_A_START)

    base_B = np.array([BASE_B_X + pRail, 0.0, 0.0])
    start_B_relative = P_B_START_WORLD - np.array([BASE_B_X, 0.0, 0.0])
    length_B = np.linalg.norm((pB - base_B) - start_B_relative)

    tA = length_A / (sA * V_ARM_A)
    tB = length_B / (sB * V_ARM_B) + abs(pRail) / (sRail * V_RAIL)
    return tA, tB


def sync_penalty(delta_t):
    if delta_t <= DELTA_T_MAX:
        return (delta_t / DELTA_T_MAX)**2
    return 1.0 + ((delta_t - DELTA_T_MAX) / DELTA_T_MAX)**2


def objective_components(x):
    x = repair_solution(x)
    tA, tB = times(x)
    delta = abs(tA - tB)
    tmax = max(tA, tB)
    psync = sync_penalty(delta)
    pfeas = feasibility_violation(x)
    J = tmax + LAMBDA_SYNC * psync + LAMBDA_FEAS * pfeas
    return {
        "J": J,
        "Tmax": tmax,
        "tA": tA,
        "tB": tB,
        "delta_t": delta,
        "Psync": psync,
        "Pfeas": pfeas,
        "feasible": is_feasible(x),
    }


def objective(x):
    return objective_components(x)["J"]


# --- wrappers opcionales para no romper imports antiguos ---

def merit_function(sol):
    """
    Wrapper para compatibilidad: recibe un 'sol' con atributos
    pA, sA, sB, sRail, pRail y construye x.
    """
    x = np.zeros(DIM)
    x[0:3] = sol.pA
    x[3] = sol.sA
    x[4] = sol.sB
    x[5] = sol.sRail
    x[6] = sol.pRail
    comp = objective_components(x)
    return comp["J"], comp["tA"], comp["tB"]


def evaluate_global_cost(sol):
    J, tA, tB = merit_function(sol)
    T_max = max(tA, tB)
    return T_max, J
