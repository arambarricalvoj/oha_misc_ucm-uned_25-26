import numpy as np
from spatialmath import SE3
from ssik.prebuilt import xarm6_ik
from kinematics_xarm6 import manipulability_xarm6

# ssik: solver analítico preconstruido
ikA_solver = xarm6_ik
ikB_solver = xarm6_ik


def _extract_q_list(sols):
    if sols is None:
        return []
    q_list = []
    for s in sols:
        if hasattr(s, "q"):
            q_list.append(np.array(s.q, dtype=float))
    return q_list


def ik_A(pA, R_A, q_seed):
    T = SE3(pA) * SE3.RPY(R_A)
    sols = ikA_solver.solve(T.A)
    q_list = _extract_q_list(sols)
    if len(q_list) == 0:
        return None, False
    q_arr = np.vstack(q_list)
    idx = np.argmin(np.linalg.norm(q_arr - q_seed, axis=1))
    return q_arr[idx], True


def ik_B(pB, R_B, q_seed, pRail):
    baseB = SE3(pRail, 0, 0)
    T = baseB.inv() * (SE3(pB) * SE3.RPY(R_B))
    sols = ikB_solver.solve(T.A)
    q_list = _extract_q_list(sols)
    if len(q_list) == 0:
        return None, False
    q_arr = np.vstack(q_list)
    idx = np.argmin(np.linalg.norm(q_arr - q_seed, axis=1))
    return q_arr[idx], True


def manipulability(robot, q):
    return manipulability_xarm6(q)
