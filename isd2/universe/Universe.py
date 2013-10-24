"""
TODO: documentation missing
"""
from isd2.universe.AtomCollection import AtomCollection

## TODO: cPickle from csb

import cPickle, numpy as np

class Universe(AtomCollection):
    """
    The Universe holds all atoms of the molecular system. Therefore
    whenever an Atom is created, it is automatically added to the universe.
    The universe is implemented using the Singleton pattern. In addition
    to the atoms it stores a N x 3 dimensional arrays for the coordinates
    and forces. These arrays are used to evaluate energies, restraints,
    compute gradients, etc.
    """
    _instance = None
    
    @staticmethod
    def get():
        """
        Get the global L{Universe} instance.
        
        This is a static Universe factory method, which makes the singleton 
        behavior more obvious (self-documentation).
        """
        if Universe._instance is None:
            Universe._instance = Universe()
        
        return Universe._instance      

    def __init__(self):
        
        if Universe._instance is not None:
            msg = "Can't create more than one Universe. Try Universe.get()"
            raise NotImplementedError(msg)

        super(Universe, self).__init__()
        self._coordinates = np.zeros((0,3))
        self._forces = np.zeros((0,3))

        Universe._instance = self

    def reset(self):
        Universe._instance = None
        Universe.__init__(self)

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def forces(self):
        return self._forces

    def add_atom(self, atom):

        atom.index = int(len(self))

        self._atoms.append(atom)
        self._coordinates = np.append(self._coordinates, np.zeros((1,3)), axis=0)
        self._forces = np.append(self._forces, np.zeros((1,3)), axis=0)

    def del_atom(self, atom):

        self._atoms.remove(atom)
        self._coordinates = np.append(self._coordinates[:atom._index],
                                      self._coordinates[atom._index+1:], axis=0)
        self._forces = np.append(self._forces[:atom._index],
                                 self._forces[atom._index+1:], axis=0)
        
        for i in range(atom._index, len(self)):
            self._atoms[i].index = self._atoms[i].index - 1

    def __reduce__(self):
        return (UniverseFactory(), (), tuple(self))

    def __setstate__(self, state):
        pass

    def dump(self, fn):
        with open(fn,'w') as stream:
            atoms = tuple(self.atoms)
            cPickle.dump((atoms, self.coordinates, self.forces), stream, 2)

    @staticmethod
    def load(fn):

        factory = AtomFactory()

        with open(fn) as stream:
            atoms, coordinates, forces = cPickle.load(stream)

            ## We don't need to explicitely create the atoms with the factory
            ## because unpickling of the atoms does the job already
            ## for atom in atoms: factory(atom.name, **atom.__dict__)

        universe = Universe.get()
        universe._coordinates[-len(atoms):,:] = coordinates
        universe._forces[-len(atoms):,:] = forces

        return universe

class UniverseFactory(object):
    
    def __call__(self):

        universe = Universe.get()

        return universe

