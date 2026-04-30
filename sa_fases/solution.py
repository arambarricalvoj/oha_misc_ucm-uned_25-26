import copy
import numpy as np


class Solution:
    def __init__(self):
        # Nivel 1
        self.pA = None
        self.pB = None
        self.rail_min = None
        self.rail_max = None
        self.pRail = None

        # Nivel 2
        self.sA = 0.5
        self.sB = 0.5
        self.sRail = 0.5

        # Orientaciones (para más adelante)
        self.qA = None
        self.qB = None

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return (
            "Solution(\n"
            f"  pA={self.pA},\n"
            f"  pB={self.pB},\n"
            f"  rail_min={self.rail_min}, rail_max={self.rail_max},\n"
            f"  pRail={self.pRail},\n"
            f"  sA={self.sA}, sB={self.sB}, sRail={self.sRail}\n"
            ")"
        )
