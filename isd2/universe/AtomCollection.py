"""
TODO: documentation missing
"""
from isd2.core.Viewable import Viewable

import numpy as np

class AtomCollection(Viewable):
    """
    An AtomCollection is a class to group atoms logically and
    provides convenient set and get routines for the coordinates
    and forces of the atom set. This class could be used to
    define rigid groups or chain molecules with internal degrees
    of freedom. 

    AtomCollection can be viewed with a StructureView and is
    therefore a subclass of Viewable
    """
    _max_printable = 7
    
    def __init__(self, atoms=None):

        super(AtomCollection, self).__init__()

        if atoms is None: atoms = []
        self._atoms = list(atoms)
        self._indices = None

    def __iter__(self):
        return iter(self._atoms)

    def __len__(self):
        return len(self._atoms)

    def __getitem__(self, index):
        return self._atoms[index]

    @property
    def indices(self):
        if self._indices is None:
            self._indices = np.array([a._index for a in self])
        return self._indices

    @property
    def coordinates(self):

        from isd2.universe.Universe import Universe
        
        universe = Universe.get()
        return universe.coordinates[self.indices]

    @coordinates.setter
    def coordinates(self, coordinates):

        assert len(coordinates) == len(self)

        from isd2.universe.Universe import Universe
        from isd._isd import set_coordinates3d

        universe = Universe.get()
        set_coordinates3d(universe.coordinates, self.indices, coordinates)

        self.changed()

    @property
    def forces(self):
        from isd2.universe.Universe import Universe
        universe = Universe.get()
        return universe.forces[self.indices]

    ## TODO: do we need this???
    ## @coordinates.setter
    ## def forces(self, forces):

    ##     assert len(forces) == len(self)

    ##     from isd.future.Universe import Universe
    ##     from isd._isd import set_coordinates3d

    ##     universe = Universe.get()
    ##     set_coordinates3d(universe.forces, self.indices, forces)

    @property
    def atoms(self):
        return self._atoms

    def __str__(self):

        if len(self) > self._max_printable:
            atoms = self._atoms[:self._max_printable/2]
            atoms+= ['...,']
            atoms+= self._atoms[-(self._max_printable/2):]
        else:
            atoms = self._atoms
            
        return '%s:\n ' % self.__class__.__name__ + '\n '.join(map(str, atoms))

    def is_empty(self):
        return len(self) == 0

