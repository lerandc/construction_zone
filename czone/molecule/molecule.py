from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms
from ..transform import BaseTransform

import copy

class BaseMolecule(ABC):
    """Base abstract class for Molecule objects.

    Molecule objects are intended to facilitate molecular atomic items, which
    are not easily generated in the Generation-Volume pair scheme. They are also
    intended to facilitate applications, for example, in surface chemistry studies.
    The molecule class mostly interfaces with other packages more suited for molecular
    generation. 

    BaseMolecules are typically not created directly.

    Attributes:
        atoms (np.ndarray): Nx3 array of atom positions of atoms in molecule
        species (np.ndarray): Nx1 array of atomic numbers of atom in molecule
        origin (np.ndarray): Reference origin of molecule.
        orientation (np.ndarray): Reference orientation of molecule.
        priority (int): Relative generation precedence of molecule.
        ase_atoms (Atoms): Collection of atoms in molecule as ASE Atoms object
    
    """

    @abstractmethod
    def __init__(self, species=None, positions=None, **kwargs) -> None:
        self._atoms = None
        self._species = None
        self.priority = 0
        self.reset_orientation()
        self._print_warnings = True

        # both species and positions must be provided
        set_check0 = (species is not None) or (positions is not None)
        set_check1 = (species is not None) and (positions is not None)
        if set_check0 and set_check1:
            self.set_atoms(species, positions)

        if "orientation" in kwargs.keys():
            self.orientation = kwargs["orientation"]

    @property
    def print_warnings(self):
        return self._print_warnings

    @print_warnings.setter
    def print_warnings(self, val):
        self._print_warnings = val

    @property
    def atoms(self):
        """Array of atomic positions of atoms lying within molecule."""
        return self._atoms

    @property
    def species(self):
        """Array of atomic numbers of atoms lying within molecule."""
        return self._species

    def set_atoms(self, species, positions):
        # check size compatibilities; cast appropriately; set variables
        species = np.array(species)
        species = np.reshape(species, (-1,)).astype(int)
        positions = np.array(positions)
        positions = np.reshape(positions, (-1, 3))

        assert(positions.shape[0] == species.shape[0])

        self._species = species
        self._atoms = positions

    def update_positions(self, positions):
        positions = np.array(positions)
        positions = np.reshape(positions, self.positions.shape).astype(int)
        self._atoms = positions

    def update_species(self, species):
        species = np.array(species)
        species = np.reshape(species, self.species.shape).astype(int)
        self._species = species

    def remove_atoms(self, indices, new_origin_idx=None):
        # check to see if origin index in removal indices
        # if so, set new origin index to 0
        # if not, update origin index appropriately
        # create copies of species and atoms arrays and remove and validate sizes

        if self._origin_tracking and self._origin_idx in indices:
            self._origin_idx = 0 if new_origin_idx is None else new_origin_idx

        self._species = np.delete(self.species, indices, axis=0)
        self._atoms = np.delete(self.atoms, indices, axis=0)
        return

    @property
    def ase_atoms(self):
        """Collection of atoms in molecule as ASE Atoms object."""
        return Atoms(symbols=self.species, positions=self.atoms)

    @property
    def origin(self):
        if self._origin_tracking:
            return self.atoms[self._origin_idx,:]
        else:
            return self._origin

    @property
    def _origin_tracking(self) -> bool:
        return self.__origin_tracking

    @_origin_tracking.setter
    def _origin_tracking(self, val: bool):
        assert(isinstance(val, bool))

        self.__origin_tracking = val

    @property
    def _origin_idx(self) -> int:
        return self.__origin_idx

    @_origin_idx.setter
    def _origin_idx(self, val: int):
        assert(isinstance(val, int))
        assert(val < self.atoms.shape[0])

        self.__origin_idx = val

    @property
    def priority(self):
        """Relative generation precedence of molecule."""
        return self._priority

    @priority.setter
    def priority(self, priority):
        if not isinstance(priority, int):
            raise TypeError("Priority needs to be integer valued")

        self._priority = priority

    def transform(self, transformation: BaseTransform, transform_origin=True):
        """Transform molecule with given transformation.

        Args:
            transformation (BaseTransform): transformation to apply to molecule.
        """
        assert (isinstance(transformation, BaseTransform)
               ), "Supplied transformation not transformation object."

        self.set_atoms(self.species, transformation.applyTransformation(self.atoms))

        if transform_origin:
            if self._origin_tracking:
                if self.print_warnings:
                    print("Requested to transform molecule, but currently origin is set to track an atom. \n \
                        Origin will not be transformed.")
                    print("Molecule is currently tracking origin against atom %i." % self._origin_idx)
                return
            self.set_origin(point=transformation.applyTransformation(self.origin))
        

    def set_origin(self, point=None, idx=None) -> None:
        """Set the reference origin to global coordinate or to track specific atom.

        Args:
            point (np.ndarray): 
            idx (int):
        """
        if point is not None:
            point = np.array(point).ravel()
            assert(point.shape == (3,))
            self._origin_tracking = False
            self._origin = point

        elif idx is not None:
            self._origin_tracking = True
            self._origin_idx = idx

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, mat):
        # check for valid rotation matrix
        # rotation matrix transforms zone axes to global coordinate system
        assert (mat.shape == (3,
                              3)), "Input matrix must be square 3x3 numpy array"
        assert (np.abs(np.linalg.det(mat) - 1.0) < 1e-6
               ), "Input matrix not a valid rotation matrix. Fails determinant."
        assert (
            np.sum(np.abs(mat @ mat.T - np.eye(3))) < 1e-6
        ), "Input matrix not a valid rotation matrix. Fails orthogonality."

        self._orientation = mat


    def reset_orientation(self):
        """Reset orientation to align with global XYZ. Does not transform molecule."""
        self.orientation = np.eye(3)

    def populate_atoms(self):
        # return self.atoms
        pass

    def checkIfInterior(self, testPoints: np.ndarray):
        ## TODO
        # have a minimum bond distance
        # perhaps set heuristically to maximum atomic radius for any of the constiuent atoms?
        return np.zeros(testPoints.shape[0], dtype=bool)

    @classmethod
    def from_ase_atoms(cls, atoms):
        ## TODO
        return

    @classmethod
    def from_pmg_molecule(cls, atoms):
        ## TODO
        return

    def from_molecule(self, **kwargs):
        """Constructor for new Molecules from existing Molecule object

        Args:
            **kwargs: "transformation"=List[BaseTransformation] to apply a 
                        series of transformations to the copied molecule.
        """

        new_molecule = copy.deepcopy(self)

        if "transformation" in kwargs.keys():
            for t in kwargs["transformation"]:
                new_molecule.transform(t)

        return new_molecule

class Molecule(BaseMolecule):
    """Standard object for representing molecules.
    
    """

    def __init__(self, species, positions, **kwargs):
        super().__init__(species, positions, **kwargs)
