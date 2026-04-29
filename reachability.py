import numpy as np
from spatialmath import SE3


def is_reachable_world(robot, p_world, tol=1e-3):
    """
    Comprueba si un punto en el mundo es alcanzable por el robot.
    Asume que robot.base ya está en el marco del mundo.
    """

    T_target = SE3(p_world[0], p_world[1], p_world[2])

    sol = robot.ikine_LM(
        T_target,
        q0=np.zeros(robot.n),
        ilimit=100,
        tol=tol
    )

    if not sol.success:
        return False, None

    # Verificación
    T_sol = robot.fkine(sol.q)
    p_sol = T_sol.t
    dist = np.linalg.norm(p_sol - p_world)

    return dist < tol, sol.q
