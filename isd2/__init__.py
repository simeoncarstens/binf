"""
Future ISD stuff goes here. Currently, the major focus is to redesign the
Universe and access to atoms, molecules etc. Also Posterior and ISDSampler
will be redesigned at some point.
"""
__version__ = '2.0.0'

from abc import ABCMeta, abstractmethod

import numpy as np, sys

sys.setrecursionlimit(int(1e6))

## TODO: will we need this at all?
class isdobject(object):

    import numpy as np

DEBUG = True


class AbstractISDNamedCallable(object):

    __metaclass__ = ABCMeta

    def __init__(self, name):

        self._name = name
        self._variables = set()
        
    def _register_variable(self, name):

        if type(name) != str:
            raise ValueError('Variable name must be a string, not ' + type(name))
        elif name in self._variables:
            raise ValueError('Variable name \"' + name + '\" must be unique')
        else:
            self._variables.add(name)
  
    def _delete_variable(self, name):

        if name in self._variables:
            self._variables.remove(name)
        else:
            raise ValueError('\"' + name + '\": unknown variable name')

    @property
    def variables(self):
        return self._variables

    @property
    def name(self):
        return _name

    @abstractmethod
    def __call__(self, **variables):
        pass


class AbstractISDNamedDifferentiableCallable(AbstractISDNamedCallable):

    @abstractmethod
    def gradient(self, **variables):
        pass
