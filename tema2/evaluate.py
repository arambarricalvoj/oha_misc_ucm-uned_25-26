import numpy as np
from robots import ik_A, ik_B, manipulability

# Velocidades nominales (ajusta a tu caso real)
V_RAIL = 1.0
V_ARM_A = 1.0
V_ARM_B = 1.0

# El raíl de B parte en su posición máxima
PRAIL_START = 0.7

# Parámetros de penalización (ejemplo)
D_EF = 0.20      # distancia objetivo en X
EPS_EJE = 0.01
EPS_Y = 0.01
EPS_Z = 0.01
DELTA_T_MAX = 0.5


def compute_times(sol):
    """
    t_r = dist_rail / (s_rail v_rail) + l_r / (s_r v_arm_r)
    A: sin raíl
    B: raíl desde PRAIL_START hasta sol.pRail
    """
    
    # Robot A
    pA = sol.pA
    lA = np.linalg.norm(pA)
    tA = lA / (sol.sA * V_ARM_A + 1e-9)

    # Robot B
    pB = sol.pB
    baseB = np.array([0.75 + sol.pRail, 0.0, 0.0])
    lB = np.linalg.norm(pB - baseB)

    dist_rail_B = abs(sol.pRail - PRAIL_START)
    tRail = dist_rail_B / (sol.sRail * V_RAIL + 1e-9)

    tB = tRail + lB / (sol.sB * V_ARM_B + 1e-9)

    return tA, tB


"""
def compute_times(sol):
    # Robot A
    qA = sol.qA
    lA = np.linalg.norm(qA - np.zeros_like(qA))
    tA = lA / (sol.sA * V_ARM_A + 1e-9)

    # Robot B
    qB = sol.qB
    lB = np.linalg.norm(qB - np.zeros_like(qB))

    dist_rail_B = abs(sol.pRail - PRAIL_START)
    tRail = dist_rail_B / (sol.sRail * V_RAIL + 1e-9)

    tB = tRail + lB / (sol.sB * V_ARM_B + 1e-9)

    return tA, tB
"""

def penalty_separation(sol):
    eje = np.array([1.0, 0.0, 0.0])
    projA = eje @ sol.pA
    projB = eje @ sol.pB
    err = abs((projA - projB) - D_EF) - EPS_EJE
    return max(0.0, err)**2


def penalty_alignment(sol):
    diff = sol.pA - sol.pB
    err_y = abs(diff[1]) - EPS_Y
    err_z = abs(diff[2]) - EPS_Z
    py = max(0.0, err_y)**2
    pz = max(0.0, err_z)**2
    # Término de orientación pendiente (cuando metas qA, qB)
    return py + pz


"""def penalty_sync(tA, tB):
    err = abs(tA - tB) - DELTA_T_MAX
    return max(0.0, err)**2
"""

def penalty_sync(tA, tB, delta_max=0.5):
    dt = abs(tA - tB)

    if dt <= delta_max:
        # Dentro del margen: penalización suave
        return (dt / delta_max)**2
    else:
        # Fuera del margen: penalización fuerte
        return 1.0 + ((dt - delta_max) / delta_max)**2

def penalty_singularity(sol, m_min=0.05):
    qA, okA = ik_A(sol.pA, sol.RA, np.zeros(6))
    qB, okB = ik_B(sol.pB, sol.RB, np.zeros(6), sol.pRail)

    if not okA or not okB:
        return 1e6

    mA = manipulability('A', qA)
    mB = manipulability('B', qB)

    # Evitar división por cero
    eps = 1e-6
    mA = max(mA, eps)
    mB = max(mB, eps)

    return (m_min / mA)**2 + (m_min / mB)**2
    

"""def penalty_singularity(sol):
    # Placeholder: cuando metas IK + Jacobiano, lo activas aquí
    return 0.0
"""

"""
def penalty_singularity(sol):
    qA, successA = ik_A(sol.pA, sol.qA)
    qB, successB = ik_B(sol.pB, sol.qB, sol.pRail)

    if not successA or not successB:
        return 1e6  # penalización enorme

    mA = manipulability(robotA, qA)
    mB = manipulability(robotB, qB)

    return max(0, m_min - mA)**2 + max(0, m_min - mB)**2
"""


def merit_function(sol):
    """
    J_total = P_sep + P_alineación + P_sync + P_sing
    (sin T_max, según tu definición revisada)
    """
    tA, tB = compute_times(sol)

    p_sep = penalty_separation(sol)
    p_align = penalty_alignment(sol)
    p_sync = penalty_sync(tA, tB)
    p_sing = 0.0 #penalty_singularity(sol)

    J = p_sep + p_align + p_sync + p_sing
    return J, tA, tB


def evaluate_global_cost(sol):
    """
    Devuelve T_max y J_total para comparar soluciones.
    SA de nivel 2 usará J_total; tú puedes usar T_max para ranking final.
    """
    J, tA, tB = merit_function(sol)
    T_max = max(tA, tB)
    return T_max, J
