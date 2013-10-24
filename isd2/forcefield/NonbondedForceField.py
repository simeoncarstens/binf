from isd2.StructureView import StructureView
from isd2.DistanceMatrix import ThresholdedDistanceMatrix as DistanceMatrix

import numpy as np

class ForceField(StructureView):

    def __init__(self, atoms, cutoff=3.71):

        super(ForceField, self).__init__(atoms)

        self._values = DistanceMatrix(len(atoms),len(atoms),cutoff)
        self._k = None
        self._d = None
        self._types = None
        
        self.set_ff_params()

    @property
    def k(self):
        """
        Matrix of pairwise force constants

        It there are N different atom types, this is a N x N matrix
        """
        return self._k

    @property
    def d(self):
        """
        Matrix of sums of van der Waals radii

        It there are N different atom types, this is a N x N matrix
        """        
        return self._d

    @property
    def types(self):
        """
        Numeric array of rank (#atoms,) storing the atom types of all
        atoms
        """
        return self._types

    def set_ff_params(self):

        n = 23

        self._k = np.ones((n,n)) 
        self._d = np.ones((n,n)) * self._values.threshold

        self._types = np.zeros(len(self._atoms))

    def update(self):

        self._values.update(self._atoms.coordinates, self._atoms.coordinates)

    def energy(self):

        distances = self.get()

        E = 0.

        for i, j, d in distances:
            if abs(i-j) < 3: continue
            k = self.k[self.types[i], self.types[j]]
            r = self.d[self.types[i], self.types[j]]
            E += k * (d-r)**4

        return E

if __name__ == '__main__':

    from isd2.universe import universe_from_pdbentry
    from isd2.universe.AtomCollection import AtomCollection
    from isd2.universe.Universe import Universe

    from scipy.spatial.distance import pdist, squareform
    
    code = '1UBQ'
    universe = Universe.get()
    if universe.is_empty(): universe = universe_from_pdbentry(code,chains=['A'])
    calphas = [a for a in universe if a.name == 'CA']
    calphas.sort(lambda a, b: cmp(a.residue.sequence_number, b.residue.sequence_number))
    calphas = AtomCollection(calphas)

    threshold = 7.5
    ff = ForceField(calphas, threshold)
    print ff.energy() * 0.5
    d = pdist(calphas.coordinates)
    n = len(calphas)
    m = d < threshold
    m*= squareform(1-np.sum([np.eye(n,k=k,dtype='i') for k in [-2, -1, 0, 1, 2]],0))
    d = np.compress(m,d)
    print np.sum((d-threshold)**4)

    from csb.core import typedproperty
    import numpy

    class arrayproperty(typedproperty):

        def __init__(self, shape, arraytype):

            super(arrayproperty, self).__init__(numpy.ndarray)
            self.shape = shape
            self.arraytype = arraytype

        def __set__(self, instance, value):
            if not isinstance(value, self.type):
                raise TypeError('expected {0}, got {1}'.format(self.type, type(value)))
            value = np.reshape(value, self.shape)
            value = value.view(self.arraytype)
            setattr(instance, self.name, value)
    
    class A(object):

        @arrayproperty((10000,),numpy.float64)
        def x():
            pass

    a = A()
    a.x = np.ones(10000)
    A.x.shape = (10,)
    
