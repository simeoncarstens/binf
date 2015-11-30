import numpy as np, os

from csb.io import tsv
from csb.bio.structure import ChemElements

from scipy.spatial import cKDTree
from scipy import sparse

class BondDefinition(object):
    """
    Stores information about covalent bonds
    bond length (in picometre) energy (in kJ/mol)
    """
    tsvfile = './covalent_bonds.tsv'

    def __init__(self):

        self._bonds = {}

        if os.path.exists(BondDefinition.tsvfile):
            self.from_tsv()

    def from_tsv(self, tsvfile=None, seps=['--','=']):

        if tsvfile is None: tsvfile = BondDefinition.tsvfile

        tab = tsv.Table.from_tsv(tsvfile)

        bonds = {}

        for bond in tab:

            atoms = bond['bond']

            for sep in seps:
                if sep in atoms:
                    atom1, atom2 = atoms.split(sep)
                    break
            else:
                continue

            elem1 = getattr(ChemElements, atom1)
            elem2 = getattr(ChemElements, atom2)
            
            ## convert length to Angstrom

            length = bond['length'] * 1e-2
            energy = float(bond['energy'])

            if not (elem1,elem2) in bonds:
                bonds[(elem1,elem2)] = {}

            bonds[(elem1,elem2)][sep] = (length, energy)

        self._bonds = bonds

    def has_bonds(self, elem1, elem2):

        return (elem1,elem2) in self._bonds or \
               (elem2,elem1) in self._bonds

    def get_bonds(self, elem1, elem2):

        if (elem1,elem2) in self._bonds:
            return self._bonds[(elem1,elem2)]

        elif (elem2,elem1) in self._bonds:
            return self._bonds[(elem2,elem1)]

        else:
            msg = 'No covalent bond definition available for ' + \
                  '{0} and {1}'.format(elem1, elem2)

            raise KeyError(msg)

    def __iter__(self):

        for (elem1, elem2), bonds in self._bonds.items():
            for type, (length, energy) in bonds.items():
                yield elem1, elem2, type, length, energy

    def __len__(self):
        return len(self._bonds)

class BondFinder(object):

    ## maximum distance between two covalently bonded atoms
    
    cutoff = 2.0

    ## maximum number of chemical bonds that an atom can form

    n_bonds = 4 + 1

    def __init__(self, bonddef=None):

        if bonddef is None: bonddef = BondDefinition()

        self._definition = bonddef

    def find_bonds(self, atoms, tol=0.1):
        """
        Returns a list of covalent bonds based on a distance criterion.
        """
        bonddef = self._definition

        elements = list(set([atom.element for atom in atoms]))
        ideal_distances = {}
        for a in elements:
            for b in elements:
                if bonddef.has_bonds(a,b):
                    d = zip(*bonddef.get_bonds(a,b).values())[0]
                    ideal_distances[(a,b)] = max(d)

        d_max = max(ideal_distances.values())

        cutoff = BondFinder.cutoff
        if cutoff < d_max + tol:
            cutoff = d_max + tol
        
        tree = cKDTree(atoms.coordinates)

        ## find nearest neighbors with kd-tree (where the number of nearest
        ## neighbors is the number of chemical bonds in biomolecules +1 for
        ## 'self-interaction')
        
        distances, indices = tree.query(atoms.coordinates,
                                        k = BondFinder.n_bonds + 1,
                                        distance_upper_bound = cutoff)

        bonds = set()

        for i in range(len(atoms)):

            a = atoms[i].element

            ## first neighbor is the atom itself

            D = distances[i][1:]
            J = indices[i][1:]
            mask = D!=np.inf
            
            for j, d in zip(J[mask],D[mask]):

                b = atoms[j].element

                if (a, b) in ideal_distances:
                    bonded = d < ideal_distances[(a,b)] + tol
                else:
                    bonded = d < cutoff                
                    print 'Bond not found:', atoms[i], atoms[j]
                    
                if bonded: bonds.add((min(i,j), max(i,j)))

        return bonds

    @classmethod
    def as_sparse_matrix(cls, atoms, bonds):
        """
        Returns a sparse connectivity matrix from an AtomCollection and
        the covalent bonds. This can also be viewed as an adjacency matrix
        of an undirected graph. 
        """
        return sparse.csr_matrix((np.ones(len(bonds),np.int8),
                                  np.transpose(list(bonds))),
                                 shape=(len(atoms),len(atoms)))

if __name__ == '__main__':

    from isd2.universe import universe_from_pdbentry
    from isd2.universe.Universe import Universe

    from scipy.sparse import csgraph
    
    universe = Universe.get()
    if universe.is_empty():
        universe = universe_from_pdbentry('1UBQ') #FNM')

    bonddef = BondDefinition()
    finder  = BondFinder(bonddef)
    bonds   = finder.find_bonds(universe)

    print '#bonds:', len(bonds)

    ## convert 

    connectivity = BondFinder.as_sparse_matrix(universe, bonds)
    
    ## find the number of bonds that separate all pairs of atoms by
    ## running the Dijkstra algorithm

    nbonds = csgraph.dijkstra(connectivity, directed=False)
    print np.sum(nbonds==1.)
    print '#{1-4 interactions}', np.sum(nbonds==4.)
    
    tree = csgraph.minimum_spanning_tree(connectivity).toarray(np.int8)
    
    from isd2.core import Node

    nodes = [Node(atom) for atom in universe]
    for i in range(len(tree)):
        for j in np.nonzero(tree[i])[0]:
            try:
                nodes[j].link(nodes[i])
            except Exception, msg:
                try:
                    nodes[i].link(nodes[j])
                except Exception, msg:
                    print msg, i, j
