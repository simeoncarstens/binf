from csb.io import tsv
from csb.bio.structure import ChemElements

from scipy.spatial import cKDTree

class BondDefinition(object):
    """
    Stores information about covalent bonds
    bond length (in pm) energy (in kJ/mol)
    """
    tsvfile = './covalent_bonds.tsv'

    def __init__(self):

        self._bonds = {}

    def from_tsv(self, tsvfile=BondDefinition.tsvfile, seps=['--','=']):

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

    def __init__(self, bonddef=None):

        if bonddef is None: bonddef = BondDefinition()

        self._definition = bonddef

    def find_bonds(self, atoms, default_length=1.5, tol=0.1, k_nearest=10):
        """
        Find covalent bonds based on a distance criterion.
        """
        bonddef = self._definition
        tree    = cKDTree(atoms.coordinates)
        cutoff  = np.ceil(max(zip(*list(bonddef))[-2])) if len(bonddef) else 0.
        cutoff  = max(cutoff, default_length)
        dist, ind = tree.query(atoms.coordinates, k=k_nearest,
                               distance_upper_bound=cutoff)

        bonds = set()

        for i in range(len(atoms)):

            a = atoms[i].element
            k = atoms[i].index
            
            for j, d in zip(ind[i],dist[i])[1:]:

                if j == len(atoms): continue

                b = atoms[j].element
                l = atoms[j].index
                
                length = default_length
                
                if bonddef.has_bonds(a, b):

                    lengths = np.array([bond[-2] for bond in
                                        bonddef.get_bonds(a, b).values()])
                    length  = lengths[np.fabs(lengths-d).argmin()]
                    
                if d < length + tol:
                    bonds.add((k,l))
                    bonds.add((l,k))

        return bonds

    @classmethod
    def as_sparse_matrix(cls, atoms, bonds):
        """
        Returns a sparse connectivity matrix from an AtomCollection and
        the covalent bonds. This can also be viewed as an adjacency matrix
        of an undirected graph. 
        """
        indices = set([atom.index for atom in atoms])

        connectivity = sparse.lil_matrix((len(atoms),len(atoms)),dtype=np.int8)

        for i in indices:

            if not i in self.bonds: continue

            j = list(set(self.bonds[i]) & indices)
            
            connectivity[i,j] = 1

        return sparse.csr_matrix(connectivity)

if __name__ == '__main__':

    from isd2.universe import universe_from_pdbentry
    from isd2.universe.Universe import Universe

    from scipy.sparse import csgraph
    
    universe = Universe.get()
    if universe.is_empty():
        universe = universe_from_pdbentry('1FNM')

    bonddef = BondDefinition()
    finder  = BondFinder(bonddef)
    bonds   = finder.find_bonds(universe, default_length=2.0, tol=0.5)

    print len(bonds)
    
    #bonds = sparse.lil_matrix((len(universe),len(universe)),dtype=np.int8)
