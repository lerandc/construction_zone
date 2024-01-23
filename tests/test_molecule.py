import unittest
import numpy as np
from czone.molecule import Molecule

seed = 8907190823
rng = np.random.default_rng(seed=seed)

class Test_Molecule(unittest.TestCase):

    def setUp(self):
        self.N_trials = 100

    def test_init(self):
        N = 1024

        ## Check basic initialization
        for _ in range(self.N_trials):
            species = rng.integers(1,119,(N,1))
            positions = rng.normal(size=(N,3))
            mol = Molecule(species, positions)

            self.assertTrue(np.allclose(mol.species, species.ravel()))
            self.assertTrue(np.allclose(mol.atoms, positions))

        ## Check input errors with wrong shaped arrays
        init_f = lambda s, p : Molecule(s,p)
        species = rng.integers(1,119,(N-1,1))
        positions = rng.normal(size=(N,3))
        self.assertRaises(ValueError, init_f, species, positions)
        
        species = rng.integers(1,119,(N,2))
        positions = rng.normal(size=(N,3))
        self.assertRaises(ValueError, init_f, species, positions)

        # Numpy should raise an error here, for the reshape of positions
        species = rng.integers(1,119,(N,1))
        positions = rng.normal(size=(N,4))
        self.assertRaises(ValueError, init_f, species, positions)

        # Reshape is valid, but the sizes are now incompatible
        N = 30
        species = rng.integers(1,119,(N,1))
        positions = rng.normal(size=(N,4))
        self.assertRaises(ValueError, init_f, species, positions)

    def test_updates(self):
        N = 1024
        for _ in range(self.N_trials):
            species = rng.integers(1,119,(N,1))
            positions = rng.normal(size=(N,3))
            mol = Molecule(species, positions)

            new_species = rng.integers(1,119,(N,1))
            new_positions = rng.normal(size=(N,3))

            mol.update_species(new_species)
            mol.update_positions(new_positions)

            self.assertTrue(np.allclose(mol.species, new_species.ravel()))
            self.assertTrue(np.allclose(mol.atoms, new_positions))

        species = rng.integers(1,119,(N,1))
        positions = rng.normal(size=(N,3))
        mol = Molecule(species, positions)
        f_update_species = lambda s: mol.update_species(s)
        f_update_positions = lambda p: mol.update_positions(p)

        bad_species = rng.integers(1,119,(N-1,1))
        self.assertRaises(ValueError, f_update_species, bad_species)

        bad_positions = rng.normal(size=(N-1,3))
        self.assertRaises(ValueError, f_update_positions, bad_positions)


    def test_removes(self):
        N = 1024
        for _ in range(self.N_trials):
            species = rng.integers(1,119,(N,1))
            positions = rng.normal(size=(N,3))
            mol = Molecule(species, positions)

            rem_ind = rng.choice(np.arange(N), 128, replace=False)
            mol.remove_atoms(rem_ind)

            keep_ind = set(np.arange(N)).difference(rem_ind)
            ref_species = np.asarray([species[i,0] for i in keep_ind])
            ref_pos = np.vstack([positions[i, :] for i in keep_ind])

            self.assertTrue(np.allclose(mol.species, ref_species))
            self.assertTrue(np.allclose(mol.atoms, ref_pos))



        species = rng.integers(1,119,(N,1))
        positions = rng.normal(size=(N,3))
        mol = Molecule(species, positions)
        bad_ind = [0, 1, 2, 3, 1025]
        self.assertRaises(IndexError, mol.remove_atoms, bad_ind)

        bad_ind = [0, 1, 2, 3, -1025]
        self.assertRaises(IndexError, mol.remove_atoms, bad_ind)

    def test_ase_atoms(self):
        N = 1024
        for _ in range(self.N_trials):
            species = rng.integers(1,119,(N,1))
            positions = rng.normal(size=(N,3))
            mol = Molecule(species, positions)

            ase_mol = mol.ase_atoms
            self.assertTrue(np.allclose(mol.species, ase_mol.get_atomic_numbers()))
            self.assertTrue(np.allclose(mol.atoms, ase_mol.get_positions()))
