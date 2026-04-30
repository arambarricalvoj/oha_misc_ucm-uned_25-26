import numpy as np

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

class Solution:
    def __init__(self, pA, qA, sA, sB, sRail):
        self.pA = np.array(pA, dtype=float)
        self.qA = np.array(qA, dtype=float)
        self.sA = float(sA)
        self.sB = float(sB)
        self.sRail = float(sRail)
        self.update_dependent()

    def update_dependent(self):
        # Offset entre efectores (ajústalo si tu problema usa otro)
        offset = np.array([0.0, 0.20, 0.0])
        self.pB = self.pA + offset

        # Rotación 180º alrededor de Z
        q180 = np.array([0.0, 0.0, 1.0, 0.0])
        self.qB = quaternion_multiply(self.qA, q180)

    def copy(self):
        return Solution(self.pA.copy(), self.qA.copy(),
                        self.sA, self.sB, self.sRail)
