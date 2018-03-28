"""
Tests and illustrates structure views.

Computes distances and a density map representation from a PDB entry
"""

if __name__ == '__main__':

    from isd2.universe import universe_from_pdbentry
    from isd2.universe.Universe import Universe
    from isd2.universe.AtomCollection import AtomCollection
    from isd2.structure.StructureView import PairwiseDistances, DensityMap, density_info
    
    from scipy import ndimage

    import numpy as np
    
    universe = Universe.get()
    if universe.is_empty():
        universe_from_pdbentry('1AKE',['A'])

    calphas = AtomCollection([a for a in universe if a.name == 'CA'])

    distances = PairwiseDistances(calphas)
    d = distances.get()

    resolution = 10.
    spacing = 3.
    info = density_info(calphas, resolution, spacing)

    map3d = DensityMap(calphas, info, resolution)
    map3d.update()

    calphas.coordinates = calphas.coordinates + 1000.

    d2 = distances.get()

    print np.all(d2.asarray() == d.asarray())

    if False:
        axes = [info.origin[d] + info.spacing[d] * np.arange(info.shape[d]) for d in range(3)]
        grid = axes[0][:,np.newaxis,np.newaxis] * \
               axes[1][np.newaxis,:,np.newaxis] * \
               axes[2][np.newaxis,np.newaxis,:]
        coords = np.array([(x,y,z) for x in axes[0] for y in axes[1] for z in axes[2]])
        out = ndimage.map_coordinates(calphas.coordinates,coords.T)

