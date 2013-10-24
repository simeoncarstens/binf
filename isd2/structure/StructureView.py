"""
StructureView

A View of an AtomCollection
"""

from isd2.core.View import View
from isd2.universe.AtomCollection import AtomCollection
from isd2.DistanceMatrix import PairwiseDistanceMatrix
from isd2 import DEBUG

from csb.bio.io.mrc import DensityInfo

import numpy as np

class StructureView(View):
    """
    StructureView: a class that computes simple geometric descriptors from an AtomCollection
    """
    def __init__(self, atoms):
        """
        Args:
        - atoms : Initialize with an AtomCollection
        """
        ## TODO: why restrict this to a single AtomCollection?

        if not isinstance(atoms, AtomCollection):
            msg = '{0} can only be initialized with {1}'.format(self,AtomCollection)
            raise TypeError(msg)

        super(StructureView,self).__init__(atoms)

    @property
    def _atoms(self):
        return self._object

    def update_forces(self):
        raise NotImplementedError

class PairwiseDistances(StructureView):
    
    def __init__(self, atoms):

        super(PairwiseDistances, self).__init__(atoms)

        self._values = PairwiseDistanceMatrix(len(atoms))
        
    def update(self):

        if DEBUG: print '{0}.update'.format(self.__class__.__name__)
        
        self._values.update(self._atoms.coordinates)

class DensityMap(StructureView):

    def __init__(self, atoms, info, resolution):

        super(DensityMap, self).__init__(atoms)

        from isd._isd import emmap
        self._ctype = emmap(tuple(info.shape))

        self.resolution = resolution
        self.origin = info.origin

        if type(info.spacing) is tuple:
            if not info.spacing.count(info.spacing[0]) == len(info.spacing):
                msg = "Spacing must be the same in all spatial directions"
                raise ValueError(msg)
            else:
                self.spacing = info.spacing[0]
        else:
            self.spacing = info.spacing

    @property
    def spacing(self):
        return self._ctype.spacing

    @spacing.setter
    def spacing(self, value):
        self._ctype.spacing = float(value)

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        self._resolution = float(value)
        self._ctype.width   = 0.5 * self._resolution / 3.**0.5

    @property
    def origin(self):
        return np.array(self._ctype.origin)

    @origin.setter
    def origin(self, value):
        self._ctype.origin  = tuple(map(float, value))

    @property
    def _values(self):
        return self._ctype.values

    @_values.setter
    def _values(self, value):
        pass

    @property
    def shape(self):
        return tuple(getattr(self._ctype,attr) for attr in ('nx','ny','nz'))

    def update(self):
        
        if DEBUG: print '{0}.update'.format(self.__class__.__name__)
        
        self._ctype.set_density(0.)
        self._ctype.add_density(self._atoms.coordinates)

    def write_mrc(self, filename):

        from csb.bio.io.mrc import DensityInfo, DensityMapWriter

        import os

        info = DensityInfo(np.reshape(self._values, self.shape), self.spacing, self.origin)
        writer = DensityMapWriter()
        writer.write_file(os.path.expanduser(filename), info)

def density_info(atoms, resolution, spacing, margin=3):

    lower   = atoms.coordinates.min(0)
    upper   = atoms.coordinates.max(0)
    sigma   = resolution / 2. / 3.**0.5
    margin *= sigma / spacing
    lower  -= margin
    upper  += margin

    shape = tuple(np.ceil((upper-lower)/spacing).astype('i').tolist())
    
    return DensityInfo(None, (spacing,)*3, lower, shape=shape)

