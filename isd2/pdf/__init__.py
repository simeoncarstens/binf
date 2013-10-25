"""
This module should provide general functionality and classes
related to probability density functions and their parameters.
"""
## TODO: Some this might end up in CSB at some point.

from abc import ABCMeta

from isd2 import AbstractISDNamedCallable

from csb.numeric import exp
from csb.statistics.pdf import AbstractDensity


class ParameterNotFoundError(AttributeError):

    pass


class AbstractISDPDF(AbstractDensity, AbstractISDNamedCallable):

    __metaclass__ = ABCMeta

    def __init__(self, name='', **variables):

        AbstractDensity.__init__(self)
        AbstractISDNamedCallable.__init__(self, name)

    @property
    def estimator(self):
        raise NotImplementedError
    @estimator.setter
    def estimator(self, strategy):
        pass
    def estimate(self, data):
        raise NotImplementedError

    def __call__(self, **variables):

        return exp(self.log_prob(**variables))
