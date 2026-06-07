import numpy as np

# Parámetros DH del xArm6 (consistentes con tu modelo previo)
T2_OFFSET = np.deg2rad(-79.34995)
T3_OFFSET = np.deg2rad(79.34995)
a2 = 0.28948866

d1 = 0.267
d4 = 0.3425
d6 = 0.097

a1 = 0.0
a3 = 0.0
a4 = 0.0775
a5 = 0.0
a6 = 0.076

ALPHA = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, np.pi/2, -np.pi/2])
A = np.array([a1, 0.0, a2, a4, a5, a6])
D = np.array([d1, 0.0, 0.0, d4, 0.0, d6])
OFFSET = np.array([0.0, T2_OFFSET, T3_OFFSET, 0.0, 0.0, 0.0])


def _dh_transform(theta, d, a, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,      d],
        [0.0,    0.0,     0.0,    1.0]
    ])
    return T


def fkine_xarm6(q):
    """
    Cinemática directa del xArm6.
    q: array-like de 6 articulaciones [rad]
    Devuelve T (4x4) en el marco base.
    """
    q = np.asarray(q).flatten()
    T = np.eye(4)
    for i in range(6):
        theta = q[i] + OFFSET[i]
        Ti = _dh_transform(theta, D[i], A[i], ALPHA[i])
        T = T @ Ti
    return T


def jacobian_xarm6(q):
    """
    Jacobiano geométrico 6x6 en el marco base.
    """
    q = np.asarray(q).flatten()
    T = np.eye(4)

    o_list = [T[0:3, 3].copy()]
    z_list = [T[0:3, 2].copy()]

    for i in range(6):
        theta = q[i] + OFFSET[i]
        Ti = _dh_transform(theta, D[i], A[i], ALPHA[i])
        T = T @ Ti
        o_list.append(T[0:3, 3].copy())
        z_list.append(T[0:3, 2].copy())

    o_n = o_list[-1]
    J = np.zeros((6, 6))

    for i in range(6):
        z = z_list[i]
        o_i = o_list[i]
        Jv = np.cross(z, o_n - o_i)
        Jw = z
        J[0:3, i] = Jv
        J[3:6, i] = Jw

    return J


def manipulability_xarm6(q):
    """
    Manipulabilidad = menor valor singular del jacobiano.
    """
    J = jacobian_xarm6(q)
    _, s, _ = np.linalg.svd(J)
    return float(np.min(s))
