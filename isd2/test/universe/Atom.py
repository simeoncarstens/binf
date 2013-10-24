import numpy as np

if __name__ == '__main__':

    from isd.utils import Dump, Load
    from isd2.universe.Universe import Universe
    from isd2.universe.Atom import AtomFactory
    
    factory = AtomFactory()
    atom = factory(name='CA', element='C', vector=[1,2,3])
    atom = factory(name='O', element='O', vector=[0,9,9])
    atom = factory(name='CA', element='C', vector=[9,2,3])
    atom = factory(name='O', element='O', vector=[9,9,9])
    universe = Universe.get()

    print universe.coordinates
    atom.vector = np.dot(np.random.random((3,3)), atom.vector)
    print universe.coordinates

    for atom in universe:
        print atom

    universe.del_atom(atom)

    print [a.index for a in universe]

    universe.del_atom(universe.atoms[0])

    print [a.index for a in universe]

    universe.del_atom(universe.atoms[-1])

    print [a.index for a in universe]

    for atom in universe:

        print atom, atom.vector
