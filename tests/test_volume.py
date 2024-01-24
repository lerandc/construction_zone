import unittest
import numpy as np
from czone.volume.algebraic import Sphere

seed = 32135213035
rng = np.random.default_rng(seed=seed)

class Test_Sphere(unittest.TestCase):
    def setUp(self):
        self.N_trials = 100
        self.N_test_points = 1024

    def test_init(self):
        for _ in  range(self.N_trials):
            radius = rng.uniform(size=(1,))[0]
            center = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            sphere = Sphere(radius, center, tol=tol)

            self.assertEqual(radius, sphere.radius)
            self.assertTrue(np.allclose(center, sphere.center))
            self.assertEqual(tol, sphere.tol)

            rt, ct = sphere.params
            self.assertEqual(radius, rt)
            self.assertTrue(np.allclose(center, ct))

        self.assertRaises(ValueError, lambda x: Sphere(x), 0.0)
        self.assertRaises(ValueError, lambda x: Sphere(x), -1.0)
        self.assertRaises(ValueError, lambda x, y: Sphere(x, y), 1.0, [2,0])
        self.assertRaises(ValueError, lambda x, y: Sphere(x, y), 1.0, [[2,0,0],[1,0,0,]])

    def test_check_interior(self):
        for _ in  range(self.N_trials):
            radius = rng.uniform(1, 1000, size=(1,))[0]
            center = rng.normal(0, 10, size=(3,))
            tol = rng.uniform(size=(1,))[0]

            sphere = Sphere(radius, center, tol=tol)

            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))
            ref_check = np.linalg.norm(test_points - center, axis=1) <= radius + tol

            self.assertTrue(np.array_equal(ref_check, sphere.checkIfInterior(test_points)))


# class Test_Plane(unittest.TestCase):
#     def test_b(self):
#         self.assertTrue(True)