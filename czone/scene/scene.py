"""
Scene Class
Luis Rangel DaCosta
"""

from ..volume import BaseVolume, makeRectPrism
import numpy as np
from ase import Atoms
from ase.symbols import Symbols, string2symbols
from ase.io import write as ase_write

class Scene():

    def __init__(self, bounds=None, objects=None):
        self._bounds = None
        self._objects = None
        self._checks = None

        if not(objects is None):
            if(hasattr(objects, "__iter__")):
                for ob in objects:
                    self.add_object(ob)
            else:
                self.add_object(objects)

        if bounds is None:
            #default bounding box is 10 angstrom cube
            bbox = makeRectPrism(10,10,10)
            self.bounds = np.vstack([np.min(bbox,axis=0), np.max(bbox,axis=0)])
        else:
            self.bounds = bounds

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        bounds = np.array(bounds)
        assert(bounds.shape == (2,3))
        self._bounds = bounds

    @property
    def objects(self):
        return self._objects

    def add_object(self, ob):
        #for now, only volumes are objects
        if isinstance(ob, BaseVolume):
            if self._objects is None:
                self._objects = [ob]
            else:
                self._objects.append(ob)

    @property
    def checks(self):
        return self._checks

    @property
    def all_atoms(self):
        return np.vstack([ob.atoms[self.checks[i],:] for i, ob in enumerate(self.objects)])

    @property
    def all_species(self):
        return np.hstack([ob.species[self.checks[i]] for i, ob in enumerate(self.objects)])

    @property
    def ase_atoms(self):
        cell_dims = self.bounds[1,:]-self.bounds[0,:]
        return Atoms(symbols=self.all_species, positions=self.all_atoms, cell=cell_dims)

    def species_from_object(self, idx):
        return self.objects[idx].atoms[self.checks[idx],:]

    def get_priorities(self):
        # get all priority levels active first
        self.objects.sort(key=lambda ob: ob.priority)
        plevels = np.array([x.priority for x in self.objects]) 

        # get unique levels and create relative priority array
        __, idx = np.unique(plevels, return_index=True)
        rel_plevels = np.zeros(len(self.objects)).astype(int)
        for i in idx[1:]:
            rel_plevels[i:] += 1

        offsets = np.append(idx, len(self.objects))

        return rel_plevels, offsets

    def populate(self):
        """
        First, every object populates atoms against its own boundaries

        Then, gather the list of priorities from all the objects
        For each object:
            1) Generate a True array of length ob.atoms
            2) For each object in the same priority level or lower, perform
                interiority check and repeatedly perform logical_and to see if
                atoms belong in scene

        
        -> Lower priority numbers supercede objects with high priority numbers
        -> Objects on the same priority level will not supply atoms to the scene 
            in their volume intersections
        """
        for ob in self.objects:
            ob.populate_atoms()

        rel_plevels, offsets = self.get_priorities()

        self._checks = []

        for i, ob in enumerate(self.objects):
            check = np.ones(ob.atoms.shape[0]).astype(bool)
            eidx = offsets[rel_plevels[i]+1]

            for j in range(eidx):
                if(i != j):
                    check_against = np.logical_not(self.objects[j].checkIfInterior(ob.atoms))
                    check = np.logical_and(check, check_against)

            self._checks.append(check)

    def to_file(self, fname, **kwargs):
        if "format" in kwargs.keys():
            if kwargs["format"] == "prismatic":
                dwf = set(self.all_species)
                dw_default = (0.1**2.0)*8*np.pi**2.0
                dwf = {str(Symbols([x])):dw_default for x in dwf}
                ase_write(filename=fname, images=self.ase_atoms, debye_waller_factors=dwf, **kwargs)
        else:
            ase_write(filename=fname, images=self.ase_atoms, **kwargs)