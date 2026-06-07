import numpy as np

from solution import Solution
from robots import ik_A, ik_B, manipulability
from vecinos import rail_interval_for_point

# Geometría / parámetros
D_EF = 0.20

V_RAIL = 1.0
V_ARM_A = 1.0
V_ARM_B = 1.0

PRAIL_START = 0.7
DELTA_T_MAX = 0.5
M_MIN = 0.08

RAIL_MIN = 0.0
RAIL_MAX = 0.70

LOWER = np.array([-0.20, -0.60, 0.05, 0.05, 0.05, 0.05, RAIL_MIN])
UPPER = np.array([ 0.70,  0.60, 0.95, 1.00, 1.00, 1.00, RAIL_MAX])
DIM = len(LOWER)


def pA_from_x(x):
    return x[:3]


def pB_from_x(x):
    return pA_from_x(x) + np.array([D_EF, 0.0, 0.0])


def rail_from_x(x):
    return x[6]


def speeds_from_x(x):
    return x[3], x[4], x[5]


def clip_bounds(x):
    return np.minimum(np.maximum(x, LOWER), UPPER)


def repair_solution(x):
    x = clip_bounds(x.copy())
    pB = pB_from_x(x)
    ok, rmin, rmax = rail_interval_for_point(pB, rail_min=RAIL_MIN, rail_max=RAIL_MAX)
    if ok:
        x[6] = np.clip(x[6], rmin, rmax)
    else:
        x[6] = np.clip(x[6], RAIL_MIN, RAIL_MAX)
    return clip_bounds(x)


def compute_times_from_sol(sol):
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


def penalty_sync(tA, tB, delta_max=DELTA_T_MAX):
    dt = abs(tA - tB)
    if dt <= delta_max:
        return (dt / delta_max)**2
    else:
        return 1.0 + ((dt - delta_max) / delta_max)**2


def penalty_separation(sol):
    eje = np.array([1.0, 0.0, 0.0])
    projA = eje @ sol.pA
    projB = eje @ sol.pB
    err = abs((projA - projB) - D_EF)
    return err**2


def penalty_alignment(sol):
    diff = sol.pA - sol.pB
    return diff[1]**2 + diff[2]**2


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


def build_solution_from_x(x):
    x = repair_solution(x)
    sol = Solution()
    sol.pA = pA_from_x(x)
    sol.pB = pB_from_x(x)
    sol.sA, sol.sB, sol.sRail = speeds_from_x(x)
    sol.pRail = rail_from_x(x)

    ok, rmin, rmax = rail_interval_for_point(sol.pB, rail_min=RAIL_MIN, rail_max=RAIL_MAX)
    if ok:
        sol.rail_min = rmin
        sol.rail_max = rmax
        sol.pRail = np.clip(sol.pRail, rmin, rmax)
    else:
        sol.rail_min = RAIL_MIN
        sol.rail_max = RAIL_MAX
        sol.pRail = np.clip(sol.pRail, RAIL_MIN, RAIL_MAX)

    return sol


def objective_components(x):
    x = repair_solution(x)
    sol = build_solution_from_x(x)

    tA, tB = compute_times_from_sol(sol)
    p_sync = penalty_sync(tA, tB)
    p_sep = penalty_separation(sol)
    p_align = penalty_alignment(sol)
    p_sing = penalty_singularity(sol)

    J = p_sep + p_align + p_sync + p_sing

    return {
        "J": J,
        "Tmax": max(tA, tB),
        "tA": tA,
        "tB": tB,
        "delta_t": abs(tA - tB),
        "Psync": p_sync,
        "Psep": p_sep,
        "Palign": p_align,
        "Psing": p_sing,
        "feasible": (p_sing < 1e5),
    }


def objective(x):
    return objective_components(x)["J"]


def merit_function(sol):
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


def is_feasible(x, tol=1e-6):
    comp = objective_components(x)
    return comp["feasible"]
