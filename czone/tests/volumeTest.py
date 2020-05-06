import unittest
from ..modules import Volume
import numpy as np


class Volume_Test(unittest.TestCase):

    def test_creation(self):
        testVolume = Volume()
        
    def test_convexHull(self):
        points = np.array([[0,0,0],[1,1,1],[1,0,0],[0,1,0]])
        testVolume = Volume(points=points)
        testVolume.createHull()

    def test_addPoint(self):
        points = np.array([[0,0,0],[1,1,1],[1,0,0],[0,1,0]])
        testVolume = Volume(points=points)

        #add single point
        testVolume.addPoints(np.array([0,0,1]))
        self.assertTrue(testVolume.points.shape[0]==5)

        #add multiple points
        testVolume.addPoints(np.array([[1,1,0],[0,1,1]]))
        self.assertTrue(testVolume.points.shape[0]==7)
        self.assertTrue(testVolume.points.shape[1]==3)
        self.assertTrue(len(testVolume.points.shape)==2)

    def test_addPointToEmpty(self):
        points = np.array([[0,0,0],[1,1,1],[1,0,0],[0,1,0]])
        testVolume = Volume()
        testVolume.addPoints(points)
        self.assertTrue(testVolume.points.shape == (4,3))

    def test_updateHull(self):
        #adding points should automatically update hull
        points = np.array([[0,0,0],[1,1,1],[1,0,0],[0,1,0]])
        testVolume = Volume(points=points)
        testVolume.createHull()
        testVolume.addPoints(np.array([0,0,1]))

        self.assertTrue(testVolume.hull)
