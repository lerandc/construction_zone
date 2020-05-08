import unittest
from ..volume import *
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
    
        self.assertTrue(testVolume.hull.points.shape==testVolume.points.shape)

    def test_translate(self):
        points = np.array([[0,0,0],[1,1,1],[1,0,0],[0,1,0]])
        testVolume = Volume(points=points)
        testVolume.createHull()
        
        translate_vec = np.array([0.5,0.5,0.5])
        testVolume.translate(translate_vec)
        check = np.array_equal(points,testVolume.points-translate_vec) 
        check = check and np.array_equal(testVolume.points, testVolume.hull.points)
        self.assertTrue(check)
    
    def test_collision_hulls(self):
        #if a vertex of a convex hull is interior to another hull, then they intersect
        points = makeRectPrism(1,1,1)
        volumeA = Volume(points=points)
        volumeA.createHull()
        volumeB = Volume(points=points)
        volumeB.createHull()

        #A and B do not collide
        volumeB.translate(np.array([2,0,0]))
        colliding = checkCollisionHulls(volumeA, volumeB) or checkCollisionHulls(volumeB, volumeA)
        self.assertFalse(colliding, msg="Collision detected between disjoint volumes.")

        #C is a subspace of A
        subpoints = makeRectPrism(0.5,0.5,0.5,center=np.array([0.5,0.5,0.5]))
        volumeC = Volume(points=subpoints)
        volumeC.createHull()
        colliding = checkCollisionHulls(volumeA, volumeC) and checkCollisionHulls(volumeC, volumeA) 
        self.assertTrue(colliding, msg="Subspace collision not detected.")

        #A and B intersect in one corner
        volumeB.translate(-np.array([2,0,0]))
        volumeB.translate(np.array([0.5,0.5,0.5]))
        colliding = checkCollisionHulls(volumeA, volumeB) and checkCollisionHulls(volumeB, volumeA)
        self.assertTrue(colliding, msg="Intersection collision not detected.")

        #A and B are exactly the same
        volumeB.translate(-np.array([0.5,0.5,0.5]))
        colliding = checkCollisionHulls(volumeA, volumeB) and checkCollisionHulls(volumeB, volumeA)
        self.assertTrue(colliding, msg="Identical volume collision not detected.")

        #A and B are misaligned on one axis only
        volumeB.translate(np.array([0.5,0,0]))
        colliding = checkCollisionHulls(volumeA, volumeB) and checkCollisionHulls(volumeB, volumeA)
        self.assertTrue(colliding, msg="Single axis misalignment collision not detected.")

        #A and B share a face
        #return False since volumes do not define 
        volumeB.translate(np.array([0.5,0,0]))
        colliding = checkCollisionHulls(volumeA, volumeB) or checkCollisionHulls(volumeB, volumeA)
        self.assertFalse(colliding, msg="Collision detected between disjoint volumes that share a face.")

