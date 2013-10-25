"""
This module contains all likelihoods occuring in ISD.
"""

import numpy

from abc import ABCMeta, abstractmethod

from csb.numeric import exp

from isd2 import AbstractISDNamedCallable

class AbstractLikelihood(AbstractISDNamedCallable):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name, forward_model, error_model):

        super(AbstractLikelihood, self).__init__(name)
        
        self._forward_model = forward_model
        self._error_model = error_model

    @property
    def forward_model(self):
        return self._forward_model

    @property
    def error_model(self):
        return self._error_model

    def log_prob(self, **variables):

        return self.error_model.log_prob(self.forward_model(**variables))

    def __call__(self, **variables):

        return exp(self.log_prob(**variables))


class AbstractDifferentiableLikelihood(AbstractLikelihood):

    def gradient(self, **variables):

        l = self(**variables)
        mock_data = self.forward_model(**variables)
        dfm = self.forward_model.jacobi_matrix(**variables)
        
        return numpy.dot(dfm, self.error_model.gradient(mock_data))
