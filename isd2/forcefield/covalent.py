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

def grow_tree(parent, nodes, neighbors):

    while len(neighbors[parent]):

        child = neighbors[parent].pop()
        neighbors[child].remove(parent)

        nodes[child].link(nodes[parent])

        grow_tree(child, nodes, neighbors)

def find_forest(connectivity, atoms=None):

    tree = csgraph.minimum_spanning_tree(connectivity).toarray(np.int8)

    if atoms is None: atoms = range(len(connectivity))
    
    nodes = [Node(atom) for atom in atoms]
    bonds = (tree + tree.T).astype('i')
    neigh = [np.nonzero(row)[0].tolist() for row in bonds]

    root = 0

    while 1:

        grow_tree(root, nodes, neigh)
        roots = np.nonzero(np.array(map(len, neigh)) > 1)[0]
        if len(roots) == 0: break
        root = roots[0]
    
    return [node for node in nodes if node.is_root()]

if __name__ == '__main__':

    from isd2.universe import universe_from_pdbentry
    from isd2.universe.Universe import Universe
    from isd2.core import Node

    from scipy.sparse import csgraph
    
    universe = Universe.get()
    if universe.is_empty():
        universe = universe_from_pdbentry('1AKE') #FNM')

    bonddef = BondDefinition()
    bonddef.from_tsv()

    finder = BondFinder(bonddef)
    bonds  = finder.find_bonds(universe)

    print '#bonds:', len(bonds)

    connectivity = BondFinder.as_sparse_matrix(universe, bonds)

    ## find the number of bonds that separate all pairs of atoms by
    ## running the Dijkstra algorithm

    nbonds = csgraph.dijkstra(connectivity, directed=False)
    print '#{1-4 interactions}', np.sum(nbonds==4.)
    
    ## find covalent trees
    
    forest = find_forest(connectivity, universe)

    ## collect chains

    chains = []
    
    for tree in forest:
        
        residues = list(set([node.info.residue for node in tree.get_descendants()]))
        residues.sort(lambda a, b: cmp(a.rank, b.rank))
        chains.append(residues)

    # leaves = [node for node in tree.get_descendants() if len(node.children) == 0]

    print 'define dihedrals'

    from collections import defaultdict

    dihedrals = defaultdict(set)

    for node2 in tree.get_descendants():

        if node2.is_root(): continue

        node1 = node2.parent

        nodes0 = node2.get_siblings()

        if not node1.is_root(): nodes0.insert(0, node1.parent)

        for node0 in nodes0:

            atoms = [node0, node1, node2]

            for node3 in node2.children:
                dihedrals[node2].add(tuple(atoms + [node3]))
            
    for node in dihedrals:
        print node
        print '\n'.join(['\t' + ' - '.join([str(n.info) for n in dihedral])
                         for dihedral in dihedrals[node]])

    from isd.Connectivity import load_connectivity

    topology = load_connectivity()

    for node in dihedrals:
        residue = node.info.residue
        residue_topology = topology[residue.type.name]

        found = []
        
        for dihedral in dihedrals[node]:

            atoms = [n.info.name for n in dihedral]
            for i, atom in enumerate(atoms):
                if dihedral[i].info.residue.rank == residue.rank-1:
                    atoms[i] += '-'
                elif dihedral[i].info.residue.rank == residue.rank + 1:
                    atoms[i] += '+'

            ## try to find dihedral in topology

            for name, definition in residue_topology.dihedrals.items():
                atoms2 = [definition['atom_{}'.format(i)] for i in range(1,5)]
                if atoms == atoms2:
                    break
            else:
                name = None
            found.append(name)

        name = '{0}{1}{2}'.format(node.info.residue.type.value,
                                  node.info.residue.sequence_number,
                                  node.info.name)
        print '{0}: {1}'.format(name, ', '.join(filter(None, found)))

