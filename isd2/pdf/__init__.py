"""
This module should provide general functionality and classes
related to probability density functions and their parameters.
"""
## TODO: Some of this might end up in CSB at some point.

from abc import ABCMeta

from isd2 import AbstractISDNamedCallable

from csb.numeric import exp
from csb.statistics.pdf.parameterized import ParameterizedDensity


class ParameterNotFoundError(AttributeError):

    pass


class AbstractISDPDF(ParameterizedDensity, AbstractISDNamedCallable):

    __metaclass__ = ABCMeta

    def __init__(self, name='', **variables):

        ParameterizedDensity.__init__(self)
        AbstractISDNamedCallable.__init__(self, name)

    @property
    def estimator(self):
        raise NotImplementedError
    @estimator.setter
    def estimator(self, strategy):
        pass
    
    def estimate(self, data):
        raise NotImplementedError

    def clone(self):

        from copy import copy

        return copy(self)

    def conditional_factory(self, **fixed_vars):

        from copy import copy
        
        result = self.clone()

        ## Does this break encapsulation?
        
        for v in fixed_vars:
            result._delete_variable(v)
            result._register(v)
            result[v] = fixed_vars[v]

        result.log_prob = lambda **variables: self.log_prob(**dict(variables, **fixed_vars))
        result.__call__ = lambda **variables: self.__call__(**dict(variables, **fixed_vars))

        return result

    def __call__(self, **variables):

        return exp(self.log_prob(**variables))
