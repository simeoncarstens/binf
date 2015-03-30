"""
Classes to compute complete and incomplete distance matrices.
"""

## TODO: think about using MaskedArrays or sparse matrices
## TODO: move test code somewhere else

import numpy as np
from scipy import spatial

class DistanceMatrix(object):
    """
    Full Euclidean distance matrix. 
    """
    def __init__(self, M, N):

        self._shape = (int(M), int(N))
        self._init_values()
        
    def _init_values(self):
        
        self._values = np.zeros(self.shape)

    @property
    def shape(self):
        return self._shape

    def update(self, X, Y):

        if not (len(X),len(Y)) == self.shape:
            raise ValueError

        self._values[:,:] = spatial.distance.cdist(X, Y)

    def get(self):
        """
        returns the raw values
        """
        return self._values

    def asarray(self):

        return self._values

    def __iter__(self):

        M, N = self.shape

        for i in range(M):
            for j in range(N):
                yield i, j, self._values[i,j]

class PairwiseDistanceMatrix(DistanceMatrix):

    def __init__(self, N):

        super(PairwiseDistanceMatrix,self).__init__(N, N)

    def _init_values(self):
        N = len(self)
        self._values = np.zeros(N*(N-1)/2)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):

        M, N = self.shape

        k = 0
        for i in range(M):
            for j in range(i+1,N):
                yield i, j, self._values[k]
                k += 1
                
    def asarray(self):
        return spatial.distance.squareform(self._values)
        
    def __getitem__(self, i, j):
        raise NotImplementedError

    def update(self, X):

        if not len(X) == len(self):
            raise ValueError

        self._values[:] = spatial.distance.pdist(X)

class IncompleteDistanceMatrix(DistanceMatrix):
    """
    A distance matrix for certain atom pairs only
    """
    def __init__(self, M, N, pairs):
        """
        Args:
        - M, N as for DistanceMatrix
        - pairs is a list of index tuples
        """
        super(IncompleteDistanceMatrix, self).__init__(M, N)

        self._shape = (M,N)
        self._indices = np.array([]).astype('i'), np.array([]).astype('i')
        self.indices = pairs

    def _init_values(self):
        self._values = np.zeros((0,))

    @property
    def indices(self):
        return zip(*self._indices)

    @indices.setter
    def indices(self, pairs):

        if not len(pairs): return 

        i, j = zip(*pairs)
        M, N = self.shape

        if not (max(i) < M and max(j) < N):
            raise ValueError('maximum indices are {0} given {1}'.format((M-1,N-1),(max(i),max(j))))

        ## Get rid of negative indices

        i = np.array(i) % M
        j = np.array(j) % N

        ## The idea is to sort the pairs such that the distance
        ## computation is more efficient

        j = j[np.argsort(i)]
        i = np.sort(i)
        
        self._indices = i, j
        self._values = np.zeros(len(i))

    def __iter__(self):

        for i, j, d in zip(self._indices[0], self._indices[1], self._values):
            yield int(i), int(j), d

    def update(self, X, Y):

        if not (len(X), len(Y)) == self.shape:
            raise ValueError

        i, j = self._indices

        self._values[:] = np.sqrt(np.sum((X[i]-Y[j])**2,1))

    def asarray(self,sparse=False):
        """
        Invalid entries will be marked with a negative distance.
        """
        if sparse:
            from scipy import sparse
            d = sparse.lil_matrix(self.shape)
            for i, j, r in self:
                if r > 0.0:
                    d[i,j] = r
            d = sparse.csr_matrix(d)                
        else:
            d = - np.ones(self.shape)
            i, j = self._indices
            d[i,j] = self.get()

        return d 
        
    def sort_indices(self):

        if not len(self._indices[0]): return
        
        i, j = self._indices
        k = np.argsort(i)

        self._values = self._values[k]
        self._indices = i[k], j[k]

class cIncompleteDistanceMatrix(IncompleteDistanceMatrix):
    """
    Uses C distance restraints.
    """
    def sort_indices(self):
        raise NotImplementedError

    @property
    def indices(self):
        return zip(*self._indices)
    
    @indices.setter
    def indices(self, pairs):

        IncompleteDistanceMatrix.indices.fset(self,pairs)

        ## setup distance restraints

        from isd._isd import distancerestraint, noe, dataset

        noes = []
        for i, j in zip(self._indices[0],self._indices[1]+self.shape[0]):
            noes.append(noe())
            noes[-1].contributions = ((int(i),int(j)),)

        distances = dataset()
        distances.items = tuple(noes)
        distances.mock_data = self._values

        self._ctheory = distancerestraint()
        self._cdata = distances
        
    def update(self, X, Y):

        if not (len(X), len(Y)) == self.shape:
            raise ValueError

        self._ctheory.X = np.concatenate((X,Y),0)
        self._ctheory.fill_mock_data(self._cdata)

class ThresholdedDistanceMatrix(IncompleteDistanceMatrix):
    """
    A distance matrix containing only distances below a
    given threshold. Eventually, the update routine should
    make use of a non-bonded list.
    """
    def __init__(self, M, N, threshold):

        super(ThresholdedDistanceMatrix, self).__init__(M, N, [])
        self._threshold = float(threshold)

    @property
    def threshold(self):
        return self._threshold

    def update(self, X, Y):
        """
        Implementation based on kd-trees
        """
        if not (len(X),len(Y)) == self.shape:
            raise ValueError
        
        tree = spatial.cKDTree(X)
        d, i = tree.query(Y, k=len(Y), distance_upper_bound=self._threshold)
        m = d<np.inf
        j = np.compress(m.flatten(), i.flatten())
        i = np.repeat(np.arange(len(X)),np.sum(m,1))
        
        self._values = np.compress(m.flatten(), d.flatten())
        self._indices = i, j

class cThresholdedDistanceMatrix(ThresholdedDistanceMatrix):
    """
    Thresholded distance matrix using a C based non-bonded list.
    """
    def __init__(self, M, N, threshold):

        super(cThresholdedDistanceMatrix, self).__init__(M, N, threshold)

        from isd.NBGrid import NBGrid

        ## TODO: is there something to be gained if we base this on the
        ## smaller / larger array?

        self._grid = NBGrid(self.shape[0], self.threshold)

    def _init_values(self):
        ## TODO: indices and distances are all stored in _values...
        self._values = np.zeros((0,3))

    def update(self, X, Y):
        """
        Implementation based on non-bonded list
        """
        if not (len(X),len(Y)) == self.shape:
            raise ValueError

        self._grid.fill(X)
        self._values = self._grid.contacts(X,Y)

        ## TODO: this slows down the code significantly
        ##
        ## i, j, d = np.transpose(contacts)
        ## self._values = d
        ## self._indices = i.astype('i'), j.astype('i')

    def __iter__(self):
        return iter(self._values)

    def get(self):
        return np.array(self._values)[:,-1]

    @property
    def _indices(self):
        return np.transpose(self._values)[:2].astype('i')

    @_indices.setter
    def _indices(self, value):
        pass

class cThresholdedPairwiseDistanceMatrix(ThresholdedDistanceMatrix):
    """
    Thresholded pairwise distance matrix using a C based non-bonded list.
    """
    def __init__(self, N, threshold):

        super(cThresholdedPairwiseDistanceMatrix, self).__init__(N, N, threshold)

        from isd.NBList import NBList
        
        self._grid = NBList(self.threshold, 90, 1000, N).ctype

        from isd._isd import atom, universe

        atoms = []
        for i in range(N):
            atoms.append(atom())
            atoms[-1].index = i
            
        universe = universe()
        universe.atoms = tuple(atoms)
        types = np.array([a.type for a in atoms], 'i')
        universe.set_types(types)
        
        self._universe = universe
        
    def _init_values(self):
        ## TODO: indices and distances are all stored in _values...
        self._values = np.zeros((0,3))

    def update(self, X):
        """
        Implementation based on non-bonded list
        """
        if not len(X) == self.shape[0]:
            raise ValueError

        self._universe.X[:,:] = X
        self._grid.update(self._universe, 0)

    def __iter__(self):
        c, d, n = self._grid.contacts, self._grid.sq_distances, self._grid.n_contacts
        for i in range(len(c)):
            for k in range(n[i]):
                yield int(i), int(c[i,k]), float(d[i,k])

    def get(self):
        d, n = self._grid.sq_distances, self._grid.n_contacts
        return np.concatenate([d[i,:m] for i, m in enumerate(n)],0)

    @property
    def _indices(self):
        return np.transpose(self._values)[:2].astype('i')

    @_indices.setter
    def _indices(self, value):
        pass

    def asarray(self):
        """
        Invalid entries will be marked with a negative distance.
        """
        d = - np.ones(self.shape)
        i, j, v = np.array(list(self)).T
        i = i.astype('i')
        j = j.astype('i')
        v = np.sqrt(v)
        d[i,j] = v
        d[j,i] = v

        return d * (1-np.eye(self.shape[0]))
        
class SillyThresholdedDistanceMatrix(ThresholdedDistanceMatrix):

    def update(self, X, Y):
        """
        This is currently a silly implementation calculating the
        full distance matrix first
        """
        self._values = np.zeros(self.shape)
        DistanceMatrix.update(self, X, Y)
        m = (self._values <= self.threshold).flatten()
        self._values = np.compress(m, self._values.flatten())
        self._indices = np.unravel_index(np.nonzero(m)[0], self.shape)

def mytimeit(stmt):

    from time import clock
    t = clock()
    exec stmt
    return clock()-t

if __name__ == '__main__':

    M, N = 100, 1000

    X = np.random.random((M,3))
    Y = np.random.random((N,3))

    # from isd.future.Universe import universe_from_pdbentry, Universe

    # pdbcode = '1OEL'
    # pdbcode = '1FNM'
    # universe = Universe.get()
    # if not len(universe):
    #     print 'loading PDB entry', pdbcode
    #     universe = universe_from_pdbentry(pdbcode, atom_names=['CA'])

    # X = Y = universe.coordinates
    # M = N = len(universe)

    ## distance threshold
    
    threshold = 3.8
    threshold = 7.5
    
    shape = (N, N)
    
    a = DistanceMatrix(*shape)
    b = PairwiseDistanceMatrix(N)

    pairs = zip(np.random.permutation(a.shape[0])[:10],
                np.random.permutation(a.shape[1])[:10])
    c = IncompleteDistanceMatrix(a.shape[0], a.shape[1], pairs)

    e = ThresholdedDistanceMatrix(shape[0], shape[1], threshold)
    f = cThresholdedDistanceMatrix(shape[0], shape[1], threshold)
    g = SillyThresholdedDistanceMatrix(a.shape[0], a.shape[1], threshold)
    h = cThresholdedPairwiseDistanceMatrix(shape[0], threshold)

    print "full distance matrix", mytimeit("for i in range(100): a.update(Y,Y)")
    print "pairwise distance matrix", mytimeit("for i in range(100): b.update(Y)")
    print "kdTree", mytimeit("for i in range(100): e.update(Y,Y)")
    print "nonbonded list", mytimeit("for i in range(100): f.update(Y,Y)")
    print "silly", mytimeit("for i in range(100): g.update(Y,Y)")
    print "nonbonded pairwise", mytimeit("for i in range(100): h.update(Y)")
    
    c = IncompleteDistanceMatrix(a.shape[0], a.shape[1], e.indices)
    c2 = cIncompleteDistanceMatrix(a.shape[0], a.shape[1], e.indices)
    print "incomplete distance matrix", mytimeit("for i in range(100): c.update(Y,Y)")
    print "c incomplete distance matrix", mytimeit("for i in range(100): c2.update(Y,Y)")

    A = a.asarray()
    B = b.asarray()
    C = c.asarray()
    C2= c2.asarray()
    E = e.asarray()
    F = f.asarray()
    G = g.asarray()
    H = h.asarray()
    
    print 'A==B', np.all(A==B)
    M = A <= e.threshold
    E2 = M * A - (1-M)
    print np.all(E==E2), np.all(E==F), np.all(E==G), np.all(E==H), np.all(C==H), np.all(np.fabs(C2-C)<1e-10)

if False:

    import scipy.sparse.linalg
    from scipy import sparse
    
    Q = scipy.sparse.linalg.expm(c.asarray(True))

if False:
    ## testing the iterators

    for i, j, d in a:
        if j > 10: continue
        print i, j, d
        if i>0 and j>5: break

    print 
 
    for i, j, d in b:
        if j > 10: continue
        print i, j, d
        if i>0 and j>5: break

    print '\nprinting incomplete distance matrix'

    for k, (i, j, d) in enumerate(c):
        if k > 20: break
        print i, j, d, A[i,j], C[i,j]

    print np.all(C[C>=0.]==A[C>=0.])
