import numpy as np
from roboticstoolbox import DHRobot, RevoluteMDH
from spatialmath import SE3


def build_robot_arm():
    T2_offset = np.deg2rad(-79.34995)
    T3_offset = np.deg2rad(79.34995)
    a2 = 0.28948866

    L1 = RevoluteMDH(alpha=0,        a=0,      d=0.267,  offset=0)
    L2 = RevoluteMDH(alpha=-np.pi/2, a=0,      d=0,      offset=T2_offset)
    L3 = RevoluteMDH(alpha=0,        a=a2,     d=0,      offset=T3_offset)
    L4 = RevoluteMDH(alpha=-np.pi/2, a=0.0775, d=0.3425, offset=0)
    L5 = RevoluteMDH(alpha=np.pi/2,  a=0,      d=0,      offset=0)
    L6 = RevoluteMDH(alpha=-np.pi/2, a=0.076,  d=0.097,  offset=0)

    qlim_deg = np.array([
        [-360,  360],
        [-118,  118],
        [-225,   11],
        [-360,  360],
        [ -97,  180],
        [-360,  360],
    ])
    qlim_rad = np.deg2rad(qlim_deg)

    L1.qlim = qlim_rad[0]
    L2.qlim = qlim_rad[1]
    L3.qlim = qlim_rad[2]
    L4.qlim = qlim_rad[3]
    L5.qlim = qlim_rad[4]
    L6.qlim = qlim_rad[5]

    return [L1, L2, L3, L4, L5, L6]


def build_robot_A():
    arm_links = build_robot_arm()
    robotA = DHRobot(
        arm_links,
        name='RobotA',
        base=SE3(0, 0, 0)
    )
    return robotA


def build_robot_B():
    arm_links = build_robot_arm()
    robotB = DHRobot(
        arm_links,
        name='RobotB',
        base=SE3(0.75, 0, 0)
    )
    return robotB

def ik_A(pA, R_A, q_seed):
    robotA = build_robot_A()
    T = SE3(pA) * SE3.RPY(R_A)   # posición + orientación
    sol = robotA.ikine_LM(T, q0=q_seed, ilimit=100, tol=1e-4)
    return sol.q, sol.success

def ik_B(pB, R_B, q_seed, pRail):
    robotB = build_robot_B()
    baseB = SE3(pRail, 0, 0)
    T = SE3(pB) #* SE3.RPY(R_B)
    T = baseB.inv() * T   # transformar al marco del robot B
    sol = robotB.ikine_LM(T, q0=q_seed, ilimit=100, tol=1e-4)
    return sol.q, sol.success

def manipulability(robot, q):
    if robot == 'A':
        robot = build_robot_A()
    elif robot == 'B':
        robot = build_robot_B()

    J = robot.jacob0(q)
    _, s, _ = np.linalg.svd(J)
    return np.min(s)
