import unittest
from ..volume import *
import numpy as numpy

class Voxel_Test(unittest.TestCase):

    def test_creation(self):
        testVoxel = Voxel()

    def test_broadcast(self):
        #scale by scalar
        testVoxel = Voxel(scale=5)
        self.assertTrue(np.array_equal(testVoxel.sbases, 5*np.identity(3)))

        #scale by vector
        testVoxel.scale=np.array([3,4,5])
        self.assertTrue(np.array_equal(testVoxel.sbases, np.array([3,4,5])*np.identity(3)))
        
        #scale by array- need full bases to test
        bases = np.array([[1,2,3],[2,1,3],[3,1,2]])
        randScales = np.random.uniform(size=(3,3))
        testVoxel.bases = bases 
        testVoxel.scale = randScales
        self.assertTrue(np.array_equal(testVoxel.sbases, bases*randScales))

        #try to put in poorly shaped arrays
        with self.assertRaises(AssertionError):
            testVoxel = Voxel(bases=np.identity(2))
        
        with self.assertRaises(ValueError):
            testVoxel = Voxel(scale=np.identity(2))

        with self.assertRaises(AssertionError):
            testVoxel.bases = np.identity(2)

        with self.assertRaises(ValueError):
            testVoxel.scale = np.identity(2)

    def test_collinear(self):

        #test assertion on creation
        with self.assertRaises(AssertionError):
            testVoxel = Voxel(bases=np.array([[1,0,0],[1,0,0],[0,1,0]]))

        #test assertion on resetting
        testVoxel = Voxel()
        with self.assertRaises(AssertionError):
            testVoxel.bases = np.array([[1,0,0], [1,0,0], [0,1,0]])
            
        with self.assertRaises(AssertionError):
            testVoxel.bases = np.array([[1,0,0], [0,1,0], [1,0,0]])

        with self.assertRaises(AssertionError):
            testVoxel.bases = np.array([[1,0,0], [0,1,0], [0,1,0]])

    def test_origin(self):
        with self.assertRaises(AssertionError):
            testVoxel = Voxel(origin=np.array([0,0]))

if __name__ == '__main__':
    unittest.main()