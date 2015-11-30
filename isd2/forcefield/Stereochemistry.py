"""
Stereochemistry

provides information about covalent bonds (in proteins)
"""
## TODO: remove hard-wired code
## TODO: move test code to test folder
## TODO: rename Stereochemistry to CovalentGraph

from isd.Connectivity import load_connectivity
from isd2.universe.Universe import Universe
from isd2.core import Node

from scipy import sparse

from collections import defaultdict

import numpy as np

class Stereochemistry(object):

    ## TODO: hard coded

    peptide_bond = ('C', 'N')

    def __init__(self):

        self._nodes = {}
        self._bonds = defaultdict(set)

    @property
    def nodes(self):
        return self._nodes

    @property
    def bonds(self):
        return self._bonds

    def get_node(self, atom):

        if not atom in self._nodes:
            self._nodes[atom] = Node(atom)

        return self._nodes[atom]

    def link(self, atom1, atom2):
        """
        Make
        """
        self.get_node(atom2).link(self.get_node(atom1))
        self.bind(atom1, atom2)
        
    def bind(self, atom1, atom2):
        """
        Add a covalent bond between two atoms
        """
        self.bonds[atom1.index].add(atom2.index)
        self.bonds[atom2.index].add(atom1.index)
        
    def update(self, atoms):
        """
        Sets up the linking and covalent bond information. Currently, this
        only works for atoms that are part of the standard amino acids.
        """
        ## TODO: replace this part by csb.bio.nmr.AtomConnectivity

        connectivity = load_connectivity()
        
        residues = [atom.residue for atom in atoms]
        length   = len(residues)
        residues = filter(None, residues)

        if len(residues) < length:
            print 'Warning: residues missing for {0} atoms'.format(length-len(residues))

        residues = {r.rank: r for r in residues}

        for rank, residue in residues.items():

            info = connectivity[residue.type.name]

            for link in info.links:

                name1, name2 = link.atom_1, link.atom_2

                if name1 not in residue or name2 not in residue:
                    continue

                atom1, atom2 = residue[name1], residue[name2]

                if not link in info.cuts:
                    self.link(atom1, atom2)
                else:
                    self.bind(atom1, atom2)

            ## link residues

            if rank-1 in residues:

                previous = residues[rank-1]
                name1, name2 = self.peptide_bond

                if name1 in previous and name2 in residue:
                    self.link(previous[name1], residue[name2])

    def get_roots(self):
        """
        Returns all root nodes in the stereochemistry graph
        """
        return [n for n in self.nodes.values() if n.is_root()]

    def as_sparse_matrix(self, atoms=None):
        """
        Returns a sparse connectivity matrix of an AtomCollection which defaults
        to the entire Universe. This can also be viewed as an adjacency matrix of
        an undirected graph. 
        """
        atoms = atoms if atoms is not None else Universe.get()
        indices = set([atom.index for atom in atoms])
        bonds = sparse.lil_matrix((len(atoms),len(atoms)),dtype=np.int8)

        for i in indices:

            if not i in self.bonds: continue

            j = np.array(list(set(self.bonds[i]) & indices))
            
            bonds[i,j] = 1

        return sparse.csr_matrix(bonds)
    
if __name__ == '__main__':

    from isd2.universe import universe_from_pdbentry
    from isd2.universe.Universe import Universe
    from scipy.sparse import csgraph

    universe = Universe.get()
    if universe.is_empty():
        universe = universe_from_pdbentry('1UBQ')

    stereochemistry = Stereochemistry()
    stereochemistry.update(universe)

    c = stereochemistry.as_sparse_matrix()

    d = csgraph.dijkstra(c)

    ## find 1-4 interactions
    
    d_14 = d == 4.
    
