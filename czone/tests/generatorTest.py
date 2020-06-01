import unittest
from ..generator import *
from pymatgen import Structure

class Generator_Test(unittest.TestCase):

    def test_creation(self):
        testGenerator = Generator()

    def test_structure(self):
        testGenerator = Generator()
        testGenerator.BasicStructure()

    def test_simpleFill(self):
        """
        Construct a large, rectanuglar volume of 4x4x4 unit cells and fill with single unit cell
        """
        testGenerator = Generator()
        cellDims = [2.5,2.5,2.5]
        volume = []
        extents = [4,4,4]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    volume.append([i*cellDims[0]*extents[0],
                                    j*cellDims[1]*extents[1],
                                    k*cellDims[2]*extents[2]])

        print(volume)
        testGenerator.BasicStructure()
