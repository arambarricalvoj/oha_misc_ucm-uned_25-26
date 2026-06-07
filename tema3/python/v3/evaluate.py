import numpy as np
from robots import ik_A, ik_B, manipulability

V_RAIL = 1.0
V_ARM_A = 1.0
V_ARM_B = 1.0

PRAIL_START = 0.7

D_EF = 0.20
EPS_EJE = 0.01
EPS_Y = 0.01
EPS_Z = 0.01
DELTA_T_MAX = 0.5

M_MIN = 0.08


def compute_times(sol):
    pA = sol.pA
    lA = np.linalg.norm(pA)
    tA = lA / (sol.sA * V_ARM_A + 1e-9)

    pB = sol.pB
    baseB = np.array([0.75 + sol.pRail, 0.0, 0.0])
    lB = np.linalg.norm(pB - baseB)

    dist_rail_B = abs(sol.pRail - PRAIL_START)
    tRail = dist_rail_B / (sol.sRail * V_RAIL + 1e-9)

    tB = tRail + lB / (sol.sB * V_ARM_B + 1e-9)

    return tA, tB


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
    return max(0.0, err_y)**2 + max(0.0, err_z)**2


def penalty_sync(tA, tB, delta_max=0.5):
    dt = abs(tA - tB)
    if dt <= delta_max:
        return (dt / delta_max)**2
    else:
        return 1.0 + ((dt - delta_max) / delta_max)**2


def penalty_singularity(sol, m_min=M_MIN):
    qA, okA = ik_A(sol.pA, sol.RA, np.zeros(6))
    if not okA:
        return 1e6

    qB, okB = ik_B(sol.pB, sol.RB, np.zeros(6), sol.pRail)
    if not okB:
        return 1e6

    mA = manipulability('A', qA)
    mB = manipulability('B', qB)

    if mA < m_min or mB < m_min:
        return 1e6

    return (m_min / mA)**2 + (m_min / mB)**2


def merit_function(sol):
    tA, tB = compute_times(sol)

    p_sep = penalty_separation(sol)
    p_align = penalty_alignment(sol)
    p_sync = penalty_sync(tA, tB)
    p_sing = penalty_singularity(sol)

    J = p_sep + p_align + p_sync + p_sing
    return J, tA, tB


def evaluate_global_cost(sol):
    J, tA, tB = merit_function(sol)
    T_max = max(tA, tB)
    return T_max, J
