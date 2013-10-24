"""
Universe and Atoms
"""
from csb.bio.structure import Structure
from csb.bio.io.wwpdb import RemoteStructureProvider as PDB
from isd2.universe.Atom import AtomFactory
from isd2.universe.Universe import Universe

def universe_from_structure(structure, chains=None, atom_names=None):
    """
    Creates the universe from a CSB structure

    @param structure: An instance of CSB's Structure object
    @type structure: L{csb.bio.structure.Structre}

    @param chains: Optional argument that can be used to specify which
    chains will we added to the universe
    @type chains: list of strings

    @return: Returns the Universe
    @rtype: L{isd.future.Universe.Universe}

    @raise TypeError: If structure is not an instance of Structure
    """
    if not isinstance(structure, Structure):
        msg = 'First argument must be an instance of Structure'
        raise TypeError(msg)

    factory = AtomFactory()

    if chains is None: chains = structure.chains

    atoms = []
    
    for chain in chains:
        for residue in structure[chain]:
            for name in residue:
                if atom_names is not None and not name in atom_names: continue
                csbatom = residue[name]
                atom = factory(name, **csbatom.__dict__)
                residue.atoms.update(name,atom)
                atoms.append(atom)
                
    return Universe.get()

def universe_from_pdbentry(pdbcode, chains=None, atom_names=None):
    """
    Creates the universe from a PDB code

    @param pdbcode: Four letter PDB code
    @type pdbcode: string

    @param chains: Optional argument that can be used to specify which
    chains will we added to the universe
    @type chains: list of strings

    @return: Returns the Universe
    @rtype: L{isd.future.Universe.Universe}
    """
    structure = PDB().get(pdbcode)
    return universe_from_structure(structure, chains, atom_names)
