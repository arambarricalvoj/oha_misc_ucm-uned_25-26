import numpy as np
from spatialmath import SE3

# Límites del raíl
RAIL_MIN = 0.0
RAIL_MAX = 0.70

# Alcances geométricos del brazo
H_MAX = 0.762   # alcance horizontal (radio en XY)
Z_MAX = 1.029   # alcance vertical (altura máxima)


def reachable_with_rail(robot, p_world, s):
    """
    Comprueba si el punto p_world es alcanzable para un valor de raíl s.
    Modelo geométrico:
    - Alcance horizontal <= 0.762 m
    - Alcance vertical   <= 1.029 m
    """
    base = robot.base.t  # vector (x, y, z)
    base_x = base[0] + s

    # Coordenadas locales respecto a la base del robot
    p_local = p_world - np.array([base_x, base[1], base[2]])
    dx, dy, dz = p_local

    horizontal = np.sqrt(dx**2 + dy**2)

    return (horizontal <= H_MAX) and (abs(dz) <= Z_MAX)


def find_valid_rail_values(robot, p_world, samples=200):
    """
    Devuelve una lista de valores s del raíl que permitirían alcanzar p_world.
    """
    valid = []
    for s in np.linspace(RAIL_MIN, RAIL_MAX, samples):
        if reachable_with_rail(robot, p_world, s):
            valid.append(s)
    return valid


def rail_interval(robot, p_world, samples=400):
    """
    Devuelve el intervalo [s_min, s_max] donde el punto es alcanzable.
    """
    valid_s = find_valid_rail_values(robot, p_world, samples=samples)

    if len(valid_s) == 0:
        return False, None, None

    return True, min(valid_s), max(valid_s)


def ik_with_rail(robot, p_world, s):
    """
    IK del brazo para un valor fijo del raíl.
    """
    base = robot.base.t
    base_x = base[0] + s

    p_local = p_world - np.array([base_x, base[1], base[2]])
    T_local = SE3(p_local[0], p_local[1], p_local[2])

    return robot.ikine_LM(T_local)


def is_reachable_world(robot, p_world):
    """
    Devuelve:
    - reachable (bool)
    - (s, q) si hay solución válida de IK
    """

    # 1) Intentar con el raíl actual (s = 0)
    sol = ik_with_rail(robot, p_world, s=0.0)
    if sol.success:
        return True, (0.0, sol.q)

    # 2) Buscar valores de raíl válidos geométricamente
    valid_s = find_valid_rail_values(robot, p_world)

    if len(valid_s) == 0:
        return False, None

    # 3) Elegir el mínimo desplazamiento válido
    s_choice = min(valid_s)

    # 4) IK con ese valor
    sol2 = ik_with_rail(robot, p_world, s_choice)
    if sol2.success:
        return True, (s_choice, sol2.q)

    return False, None
