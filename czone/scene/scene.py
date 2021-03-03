"""
Scene Class
Luis Rangel DaCosta
"""

import numpy as np


class Scene():

    def __init__(self, bounds=None, objects=None):
        self._bounds = None
        self._all_atoms = None
        self._objects = None