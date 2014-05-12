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
        self._differentiable_variables = set()
        
    def _register_variable(self, name, differentiable=False):

        if type(name) != str:
            raise ValueError('Variable name must be a string, not ' + type(name))
        elif name in self._variables:
            raise ValueError('Variable name \"' + name + '\" must be unique')
        else:
            self._variables.add(name)
            if differentiable:
                self._differentiable_variables.add(name)
  
    def _delete_variable(self, name):

        if name in self._variables:
            self._variables.remove(name)
            if name in self.differentiable_variables:
                self._differentiable_variables.remove(name)
        else:
            raise ValueError('\"' + name + '\": unknown variable name')

    @property
    def variables(self):
        return self._variables

    @property
    def differentiable_variables(self):
        return self._differentiable_variables

    @property
    def name(self):
        return self._name

    @abstractmethod
    def __call__(self, **variables):
        
        pass

    def _check_differentiability(self, **variables):

        if len(variables.viewkeys() & set(self._differentiable_variables)) == 0:
            msg = 'Function cannot be differentiated w.r.t. any of the variables '+variables
            raise ValueError(msg)

    def _evaluate_gradient(self, **variables):

        raise NotImplementedError

    def gradient(self, **variables):

        self._complete_variables(**variables)
        result = self._evaluate_gradient(**variables)
        self._reduce_variables(**variables)

        return result
    
    def _complete_variables(self, **variables):
        '''
        _complete_variables and _reduce_variables so far only work for classes
        which both inherit from AbstractISDNamedCallable and can hold parameters
        (that is, PDFs and models)
        '''

        ## at some point, this wouldn't update the variable dict in-place when called from a log_prob. Why?
        
        variables.update(**{p: self[p].value for p in self.parameters if p in self._original_variables})

        return variables

    def _reduce_variables(self, **variables):

        for p in self.parameters:
            if p in variables:
                variables.pop(p)
