import copy
import itertools
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number

from ..transform import BaseTransform
from ..transform.strain import BaseStrain
from ..volume.voxel import Voxel
from .amorphous_algorithms import *

#####################################
########## Generator Classes ########
#####################################


class BaseGenerator(ABC):
    """Base abstract class for Generator objects. 
    
    Generator objects are additive components in Construction Zone. When designing
    nanostructures, Generators contain information about the arrangement of atoms
    in space and can supply atoms at least where they should exist. 

    BaseGenerators are typically not created directly. Use the Generator class
    for crystalline systems, and the AmorphousGenerator class for non-crystalline
    systems.
    """

    @abstractmethod
    def supply_atoms(self, bbox: np.ndarray):
        """Given a bounding region, supply enough atoms to complete fill the region.
        
        Args:
            bbox (np.ndarray): Nx3 array defining vertices of convex region

        Returns:
            Coordinates and species of atoms that fill convex region. 
            Returned as Nx3 and Nx1 arrays.
        """
        pass

    def from_generator(self, **kwargs):
        """Constructor for new Generators based on existing Generator object.

        Args:
            **kwargs: "transformation"=List[BaseTransformation] to apply a 
                        series of transformations to the copied generator.
    
        """
        new_generator = copy.deepcopy(self)

        if "transformation" in kwargs.keys():
            for t in kwargs["transformation"]:
                new_generator.transform(t)

        return new_generator


class Generator(BaseGenerator):
    """Generator object for crystal systems.

    Generator objects are additive components in Construction Zone. When designing
    nanostructures, Generators contain information about the arrangement of atoms
    in space and can supply atoms at least where they should exist. 

    The Generator class handles crystalline systems, primarily by utilizing 
    Structure and Lattice objects from pymatgen, along with an internal Voxel 
    class that maintains the state of the Generator if transformed.

    Attributes:
        structure (Structure): pymatgen Structure object encoding crystallographic
                                details.
        lattice (Lattice): Convenience property to grab lattice information from
                            Structure.
        species (np.ndarray): Atomic numbers for atoms in unit cell.
        coords (np.ndarray): Fractional coordinates of atoms in unit cell.
        orientation (np.ndarray): Orientation of generator. Not used currently.
        origin (np.ndarray): Local origin of generator in Cartesian coordinates.
        voxel (Voxel): Internal Voxel object used to span space for generator.
        strain_field (BaseStrain): Strain field applied to atoms supplied by Generator.

    """

    def __init__(self,
                 origin: np.ndarray = None,
                 structure: Structure = None,
                 strain_field: BaseStrain = None):
        self._structure = None
        self._voxel = None
        self._orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self._strain_field = None

        if not structure is None:
            self.structure = structure

        if not strain_field is None:
            self.strain_field = strain_field

        if not origin is None:
            self.origin = origin

    """
    Properties
    """

    @property
    def structure(self):
        """pymatgen Structure object encoding crystallographic details."""
        return self._structure

    @structure.setter
    def structure(self, structure: Structure):
        assert (isinstance(
            structure,
            Structure)), "Structure must be pymatgen Structure object"
        self._structure = structure

        # check if voxel exists, if so, copy origin
        if self.voxel is None:
            self.voxel = Voxel(bases=structure.lattice.matrix)
        else:
            self.voxel = Voxel(bases=structure.lattice.matrix,
                               origin=self.origin)

    @property
    def lattice(self):
        """Crystallographic lattice and unit cell information."""
        return self.structure.lattice

    @property
    def species(self):
        """Ordered list of species in unit cell."""
        return self.structure.atomic_numbers

    @property
    def coords(self):
        """Ordered list of fractional coordinates of atoms in unit cell."""
        return self.structure.frac_coords

    @property
    def orientation(self):
        """Orientation of unit cell in space. Not used."""
        return self._orientation

    @property
    def origin(self):
        """Local origin of Generator in Cartesian coordinates."""
        return self.voxel.origin

    @origin.setter
    def origin(self, val):
        if self.voxel is None:
            # create voxel to store origin for when structure is later attached
            self.voxel = Voxel(origin=val)
        else:
            self.voxel.origin = val

    @property
    def voxel(self):
        """Voxel object used to span space and keep transformed bases for Generator."""
        return self._voxel

    @voxel.setter
    def voxel(self, voxel: Voxel):
        if not isinstance(voxel, Voxel):
            raise TypeError("Supplied voxel is not of Voxel() class")

        self._voxel = voxel

    @property
    def strain_field(self):
        """Strain field applied to atoms post-generation."""
        return self._strain_field

    @strain_field.setter
    def strain_field(self, field: BaseStrain):
        assert (isinstance(
            field, BaseStrain)), "Strain field must be of BaseStrain class"
        self._strain_field = field

    """ 
    Methods
    """

    def supply_atoms(self, bbox: np.ndarray):
        #generation method for pymatgen lattice structure
        min_extent, max_extent = self.voxel.get_extents(bbox)
        fcoords = np.copy(
            self.coords)  #grab fractional in case bases are rotated

        # get matrix representing unit cells as grid
        l = [
            range(min_extent[0], max_extent[0] + 1),
            range(min_extent[1], max_extent[1] + 1),
            range(min_extent[2], max_extent[2] + 1), 
            [0]
        ]
        ucs = np.array(list(itertools.product(*l)))

        # get 4D bases with identity in final index to multiply species list
        bases = np.eye(4)
        bases[0:3, 0:3] = self.voxel.sbases
        lcoords = bases @ ucs.T
        scoords = bases @ (np.hstack([fcoords,
                                      np.array(self.species)[:, None]])).T

        coords = lcoords[:, :, None] + scoords[:, None, :]
        coords = coords.reshape(4, coords.shape[1] * coords.shape[2])
        coords = coords.T

        out_coords = coords[:, :-1] + self.voxel.origin
        species = coords[:, -1]

        if self.strain_field is None:
            return out_coords, species
        else:
            self.strain_field.scrape_params(self)
            return self.strain_field.apply_strain(out_coords), species

    def transform(self, transformation: BaseTransform):
        """Transform Generator object with transformation described by Transformation object.
        
        Args:
            transformation (BaseTransform): Transformation object from transforms module.
        """
        assert (isinstance(transformation, BaseTransform)
               ), "Supplied transformation not transformation object."
        self.voxel.bases = transformation.applyTransformation_bases(
            self.voxel.bases)

        if not (transformation.basis_only):
            new_origin = transformation.applyTransformation(
                np.reshape(self.voxel.origin, (1, 3)))
            self.voxel.origin = np.squeeze(new_origin)

    @classmethod
    def from_spacegroup(cls,
                        Z: List[int],
                        coords: np.ndarray,
                        cellDims: np.ndarray,
                        cellAngs: np.ndarray,
                        sgn: int = None,
                        sym: str = None,
                        **kwargs):
        """Convenience constructor for creating Generator with symmetric unit cell from spacegroup.
        
        Args:
            Z (List[int]): List of atomic numbers in symmetric sites of unit cell.
            coords (np.ndarray): Nx3 array of fractional coordinates of atoms
                                at symmetric sites in unit cell.
            cellDims (List[float]): Length 3 list with lattice parameters a,b,c of unit cell.
            cellAngs (List[float]): Length 3 list with lattice parameters alpha,
                                    beta, gamma of unit cell.
            sgn (int): Space group number of desired space group.
            sym (str): Space group symbol of desired space group as Full International
                        or Hermann-Mauguin symbol. Overriden by sgn, if both supplied.
            **kwargs: Any of **kwargs accepted in standard Generator construction.

        Returns:
            Generator: Generator object with structure and symmetry defined by input space group.
        """
        if sym is None:
            sg = SpaceGroup(int_symbol=sg_symbol_from_int_number(sgn))
        else:
            sg = SpaceGroup(int_symbol=sym)

        lattice = Lattice.from_parameters(a=cellDims[0],
                                          b=cellDims[1],
                                          c=cellDims[2],
                                          alpha=cellAngs[0],
                                          beta=cellAngs[1],
                                          gamma=cellAngs[2])

        structure = Structure.from_spacegroup(sg.int_number,
                                              lattice=lattice,
                                              species=Z,
                                              coords=coords)

        return cls(structure=structure, **kwargs)

    @classmethod
    def from_unit_cell(cls, Z: List[int], coords: np.ndarray,
                       cellDims: np.ndarray, cellAngs: np.ndarray, **kwargs):
        """Convenience constructor for creating Generators directly from unit cell data.

        Args:
            Z (List[int]): List of atomic numbers in symmetric sites of unit cell.
            coords (np.ndarray): Nx3 array of fractional coordinates of atoms
                                at symmetric sites in unit cell.
            cellDims (List[float]): Length 3 list with lattice parameters a,b,c of unit cell.
            cellAngs (List[float]): Length 3 list with lattice parameters alpha,
                                    beta, gamma of unit cell.
            **kwargs: Any of **kwargs accepted in standard Generator construction.

        Returns:
            Generator: Generator object with given unit cell.
        """
        tmp_lattice = Lattice.from_parameters(a=cellDims[0], b=cellDims[1], c=cellDims[2], \
                                                alpha=cellAngs[0], beta=cellAngs[1], gamma=cellAngs[2])

        structure = Structure(tmp_lattice, Z, coords)

        return cls(structure=structure, **kwargs)


class AmorphousGenerator(BaseGenerator):
    """Generator object for non-crystal systems.

    Generator objects are additive components in Construction Zone. When designing
    nanostructures, Generators contain information about the arrangement of atoms
    in space and can supply atoms at least where they should exist. 

    The AmorphousGenerator object handles non-crystalline systems.
    Currently supports only monatomic, periodically uniformly disrtributed blocks.
    Default parameters are set for the purpose of generating amorphous carbon 
    blocks.

    Attributes:
        origin (np.ndarray): Local origin of generator in Cartesian coordinates.
        species (int): Atomic number of element used in monatomic generation.
        density (float): Average density of material.
        min_dist (float): Minimum bond distance between atoms.
        old_result (np.ndarray): Coordinates of atoms in previous generation.
        use_old_result (bool): Whether or not to re-use previous generation, or 
                               to run generation routine again.

    """

    def __init__(self, origin=None, min_dist=1.4, density=.1103075, species=6):
        self._origin = None
        self._species = None
        self._density = None
        self._min_dist = None

        if not (origin is None):
            self.origin = origin

        self.species = species
        self.min_dist = min_dist
        self.density = density
        self.use_old_result = False

    """
    Properties
    """

    @property
    def origin(self):
        """Local origin of Generator in Cartesian coordinates."""
        return self._origin

    @origin.setter
    def origin(self, origin: np.ndarray):
        origin = np.array(origin)
        assert (origin.size == 3), "Origin must be a point in 3D space"
        self._origin = origin

    @property
    def species(self):
        """Atomic species used in monatomic generation."""
        return self._species

    @species.setter
    def species(self, species):
        self._species = species

    @property
    def density(self):
        """Density of atoms in material."""
        return self._density

    @density.setter
    def density(self, density):
        assert (density > 0.0)
        self._density = density

    @property
    def min_dist(self):
        """Minimum bond distance between atoms in material."""
        return self._min_dist

    @min_dist.setter
    def min_dist(self, min_dist):
        assert (min_dist > 0)
        self._min_dist = min_dist

    @property
    def old_result(self):
        """Atomic coordinates of previous generation, if supply atoms has been called."""
        return self._old_result

    """
    Methods
    """

    def supply_atoms(self, bbox):

        if self.use_old_result and self._old_result is not None:
            return self.old_result
        else:
            coords = gen_p_substrate(
                np.max(bbox, axis=0) - np.min(bbox, axis=0), self.min_dist)
            self._old_result = (coords, np.ones(coords.shape[0]) * self.species)
            return self.old_result
