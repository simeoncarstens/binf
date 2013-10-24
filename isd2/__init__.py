"""
Future ISD stuff goes here. Currently, the major focus is to redesign the
Universe and access to atoms, molecules etc. Also Posterior and ISDSampler
will be redesigned at some point.
"""
__version__ = '2.0.0'

import numpy as np, sys

sys.setrecursionlimit(int(1e6))

## TODO: will we need this at all?
class isdobject(object):

    import numpy as np

DEBUG = True
