"""
Generator Class
Luis Rangel DaCosta
"""
from .amorphous_algorithms import *
from ..transform import BaseTransform, Translation
from ..transform.strain import BaseStrain
from ..volume.voxel import Voxel
import copy
import numpy as np
from abc import ABC, abstractmethod
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number

#####################################
########## Generator Classes ########
#####################################

class BaseGenerator():

    @abstractmethod
    def supply_atoms(self):
        pass

    def from_generator(self, **kwargs):
        """
        Construct a generator from another generator
        **kwargs encodes relationship
        """
        new_generator = copy.deepcopy(self)

        if "transformation" in kwargs.keys():
            for t in kwargs["transformation"]:
                new_generator.transform(t)

        return new_generator


class Generator(BaseGenerator):
    """
    Generator class for crystal systems
    Utilizes pymatgen Lattice and Structure to supply relevant methods
    """
    def __init__(self, origin=None, structure=None, strain_field=None):
        self._structure = None
        self._voxel = None
        self._orientation = np.array([[1,0,0],[0,1,0],[0,0,1]])
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
        return self._structure

    @structure.setter
    def structure(self, structure):
        assert(isinstance(structure, Structure)), "Structure must be pymatgen Structure object"
        self._structure = structure
        
        # check if voxel exists, if so, copy origin
        if self.voxel is None:
            self.voxel = Voxel(bases=structure.lattice.matrix)
        else:
            self.voxel = Voxel(bases=structure.lattice.matrix, origin=self.origin)

    @property
    def lattice(self):
        return self.structure.lattice

    @property
    def species(self):
        return self.structure.atomic_numbers

    @property
    def coords(self):
        return self.structure.frac_coords

    @property
    def orientation(self):
        return self._orientation

    @property
    def origin(self):
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
        return self._voxel

    @voxel.setter
    def voxel(self, voxel):
        if not isinstance(voxel, Voxel):
            raise TypeError("Supplied voxel is not of Voxel() class")

        self._voxel = voxel

    @property
    def strain_field(self):
        return self._strain_field

    @strain_field.setter
    def strain_field(self, field: BaseStrain):
        assert(isinstance(field, BaseStrain)), "Strain field must be of BaseStrain class"
        self._strain_field = field

    """ 
    Methods
    """
    def supply_atoms(self, bbox):
        """
        Add atoms to bounding box region
        """
        #generation method for pymatgen lattice structure
        min_extent, max_extent = self.voxel.get_extents(bbox)
        fcoords = np.copy(self.coords) #grab fractional in case bases are rotated

        coords = []
        species = []
        # TODO: remove loops for speed
        for i in range(min_extent[0], max_extent[0]+1):
            for j in range(min_extent[1], max_extent[1]+1):
                for k in range(min_extent[2], max_extent[2]+1):
                    new_coords = np.matmul(self.voxel.sbases, np.array([i,j,k])) \
                                + np.dot(self.voxel.sbases, fcoords.T).T \
                                + self.voxel.origin
                    coords.append(new_coords)
                    species.append(self.species)

        coords = np.squeeze(np.array(coords))
        species = np.squeeze(np.array(species))

        if len(coords.shape) > 2:
            coords = np.reshape(coords, (np.prod(coords.shape[0:-1]), 3))
            species = species.ravel()

        if self.strain_field is None:
            return coords, species
        else:
            self.strain_field.scrape_params(self)
            return self.strain_field.apply_strain(coords), species

    def rotate(self,R):
        #rotate basis of voxel for equivalence
        self.voxel.rotate(R)

    def transform(self, transformation):
        assert(isinstance(transformation, BaseTransform)), "Supplied transformation not transformation object."
        self.voxel.bases = transformation.applyTransformation_bases(self.voxel.bases)

        if not (transformation.basis_only):
            new_origin = transformation.applyTransformation(np.reshape(self.voxel.origin,(1,3)))
            self.voxel.origin = np.squeeze(new_origin)

    @classmethod
    def from_spacegroup(cls, Z, coords, cellDims, cellAngs, sgn=None, sym=None, **kwargs):
        if sym is None:
            sg = SpaceGroup(int_symbol=sg_symbol_from_int_number(sgn))
        else:
            sg = SpaceGroup(int_symbol=sym)

        lattice = Lattice.from_parameters(a=cellDims[0], b=cellDims[1], c=cellDims[2],
                                            alpha=cellAngs[0], beta=cellAngs[1], gamma=cellAngs[2])

        structure = Structure.from_spacegroup(sg.int_number, lattice=lattice, species=Z, coords=coords)

        return cls(structure=structure, **kwargs)

    @classmethod
    def from_unit_cell(cls, Z=[1], coords=[[0.0, 0.0, 0.0]], cellDims=[2.5, 2.5, 2.5], cellAngs=[90, 90, 90], **kwargs):
        """
        Define simple translating unit
        """
        tmp_lattice = Lattice.from_parameters(a=cellDims[0], b=cellDims[1], c=cellDims[2], \
                                                alpha=cellAngs[0], beta=cellAngs[1], gamma=cellAngs[2])

        structure = Structure(tmp_lattice, Z, coords)

        return cls(structure=structure, **kwargs)

class AmorphousGenerator(BaseGenerator):
    """
    Currently supports only monatomic, periodically uniformly disrtributed blocks

    Defaults to carbon properties
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

    """
    Properties
    """
    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        origin = np.array(origin)
        assert(origin.size==3), "Origin must be a point in 3D space"
        self._origin = origin

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, species):
        self._species = species

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        assert(density>0.0)
        self._density = density

    @property
    def min_dist(self):
        return self._min_dist

    @min_dist.setter
    def min_dist(self, min_dist):
        assert(min_dist > 0)
        self._min_dist = min_dist
    """
    Methods
    """
    def supply_atoms(self, bbox):

        coords = gen_p_substrate(np.max(bbox,axis=0)-np.min(bbox,axis=0), self.min_dist)
        return coords, np.ones(coords.shape[0])*self.species