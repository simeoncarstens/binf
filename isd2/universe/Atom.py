from csb.bio.structure import Atom as CSBAtom
from isd2.universe.Universe import Universe

import numpy as np

class Atom(CSBAtom):
    """
    An Atom instance is always contained in the Universe.
    Atom is a subclass of Atom in csb.bio.structure. Hopefully
    we will get rid of the c-type atom.
    """
    def __init__(self, name, element=None, vector=None):
        raise NotImplementedError("Use an AtomFactory to create the Atom")

    def _init(self, name, element=None, vector=None):

        assert type(name) == str and len(name) > 0

        element = element if element is not None else name[0]
        vector = vector if vector is not None else np.zeros(3)

        super(Atom, self).__init__(self.index+1, name, element, vector)

    def clone(self):
        raise NotImplementedError("Atoms are unique")

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        index = int(index)
        assert index >= 0
        self._index = index
        
    @property
    def _vector(self):
        return Universe.get().coordinates[self._index]

    @_vector.setter
    def _vector(self, position):
        Universe.get()._coordinates[self._index,:] = position

    def __setstate__(self, state):

        for attr, value in state.items():
            setattr(self, attr, value)

    def __reduce__(self):
        """
        Be careful: since an atom might have a residue and the residue might
        be part of a chain, pickling a single atom basically pickles all atoms
        hanging-off the chain
        """
        state = self.__dict__.copy()
        args = state.pop('_name'), state.pop('_element'), self.vector
        del state['_index']
        
        return (AtomFactory(), args, state)

class AtomFactory(object):

    @staticmethod
    def __call__(name, element=None, vector=None, **kw):

        atom = object.__new__(Atom)

        universe = Universe.get()
        universe.add_atom(atom)

        atom._init(name, element, vector)

        for attr, value in kw.items():
            setattr(atom, attr, value)

        return atom

