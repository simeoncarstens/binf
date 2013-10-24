import numpy as np

if __name__ == '__main__':

    from isd2.universe.Atom import AtomFactory
    from isd2.universe.Universe import Universe
    from isd2.universe.AtomCollection import AtomCollection
    
    N = 10000
    factory = AtomFactory()
    atoms = [factory('CA') for i in range(N)]
    universe = Universe.get()
    universe._coordinates = np.random.random((N,3))

    atoms = AtomCollection([universe.atoms[i] for i in np.random.permutation(N)[:1000]])
    atoms.coordinates = np.zeros((len(atoms),3))
