"""
Scene Class
Luis Rangel DaCosta
"""

from ..volume import Volume, makeRectPrism
import numpy as np

class Scene():

    def __init__(self, bounds=None, objects=None):
        self._bounds = None
        self._objects = None

        if not(objects is None):
            if(hasattr(objects, "__iter__")):
                for object in objects:
                    self.add_object(object)
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
        assert(bounds.shape == (2,3))
        self._bounds = bounds

    @property
    def objects(self):
        return self._objects

    def add_object(self, object):
        #for now, only volumes are objects
        if isinstance(object, Volume):
            if self._objects is None:
                self._objects = [object]
            else:
                self._objects.append(object)

    @property
    def all_atoms(self):
        return np.vstack([object.atoms for object in self.objects])

    @property
    def all_species(self):
        return np.hstack([object.species for object in self.objects])

    def populate(self):
        for object in self.objects:
            object.populate_atoms()