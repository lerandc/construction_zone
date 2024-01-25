import unittest
import numpy as np
from czone.volume.algebraic import Sphere, Plane
from czone.transform import Rotation

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
            self.assertTrue(np.array_equal(center, sphere.center))
            self.assertEqual(tol, sphere.tol)

            rt, ct = sphere.params
            self.assertEqual(radius, rt)
            self.assertTrue(np.array_equal(center, ct))

        self.assertRaises(ValueError, lambda x: Sphere(x, [0, 0, 0]), 0.0)
        self.assertRaises(ValueError, lambda x: Sphere(x, [0, 0, 0]), -1.0)
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


class Test_Plane(unittest.TestCase):
    def setUp(self):
        self.N_trials = 100
        self.N_test_points = 1024

    def test_init(self):
        def is_collinear(a,b):
            cos_theta = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
            return np.isclose(cos_theta, 1.0)
        
        for _ in  range(self.N_trials):
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            plane = Plane(normal, point, tol=tol)

            self.assertTrue(np.isclose(1.0, np.linalg.norm(plane.normal)))
            self.assertTrue(is_collinear(normal, plane.normal))
            self.assertTrue(np.array_equal(point, plane.point))
            self.assertEqual(tol, plane.tol)

            nt, pt = plane.params
            self.assertTrue(np.array_equal(point, pt))
            self.assertTrue(np.isclose(1.0, np.linalg.norm(nt)))
            self.assertTrue(is_collinear(normal, nt))

            plane.flip_orientation()
            self.assertTrue(np.isclose(1.0, np.linalg.norm(plane.normal)))
            self.assertTrue(is_collinear(-normal, plane.normal))

        self.assertRaises(ValueError, lambda x: Plane(x,(0,0,0)), (0,0,0))

    def test_check_interior(self):
        for _ in  range(self.N_trials):
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            plane = Plane(normal, point, tol=tol)
            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))
            ref_check = np.dot(test_points - point, normal/np.linalg.norm(normal)) < (tol)
    
            self.assertTrue(np.array_equal(ref_check, plane.checkIfInterior(test_points)))

            # This would fail for any points on the plane or within tolerance of the interiority check
            tol_filter = plane.dist_from_plane(test_points) > plane.tol
            test_points = test_points[tol_filter, :]

            ref_check = np.dot(test_points - point, normal/np.linalg.norm(normal)) < (tol)
            ref_arr = np.logical_not(ref_check)            
            self.assertTrue(np.array_equal(ref_arr, plane.flip_orientation().checkIfInterior(test_points)))

    def test_dist_from_plane(self):
        for _ in range(self.N_trials):
            point = rng.normal(size=(3,))
            plane = Plane([0,0,1], point)
            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))

            ref_arr = np.abs(test_points[:, 2]-point[2])
            self.assertTrue(np.allclose(plane.dist_from_plane(test_points), ref_arr))

            Rs = rng.normal(size=(3,3))
            R = Rotation(np.linalg.qr(Rs)[0], origin=point)

            r_plane = R.applyTransformation_alg(plane)
            r_points = R.applyTransformation(test_points)
            self.assertTrue(np.allclose(r_plane.dist_from_plane(r_points), ref_arr))

            r_plane.flip_orientation()
            self.assertTrue(np.allclose(r_plane.dist_from_plane(r_points), ref_arr))

    def test_project_point(self):
        for _ in range(self.N_trials):
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            plane = Plane(normal, point, tol=tol)
            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))

            proj_points = plane.project_point(test_points)
            proj_dist = plane.dist_from_plane(proj_points)
            print(proj_dist[:10])
            print(test_points[3, :])
            print(plane.dist_from_plane(test_points[3,:]))
            self.assertTrue(np.allclose(proj_dist, np.zeros_like(proj_dist)))
