"""
This module contains all error models occuring in ISD.
"""

import numpy

from abc import ABCMeta, abstractmethod

from isd2.model import AbstractModel
from isd2.pdf import ParameterNotFoundError, AbstractISDPDF

from csb.numeric import log, exp


class AbstractErrorModel(AbstractISDPDF, AbstractModel):

    __metaclass__ = ABCMeta
    

class GaussianErrorModel(AbstractErrorModel):

    def __init__(self, name, sigma):

        super(GaussianErrorModel, self).__init__(name)

        self._register('sigma')
        self['sigma'] = sigma

        self._register_variable('mock data')

    def log_prob(self, mock_data):
        """
        Currently hacked and tailored for the example in sandbox_simeon.py.
        We need to think about a MockData class
        """
        
        ss = self['sigma'].value ** 2
        single_probs = exp(-0.5 * (mock_data[2] - mock_data[1]) ** 2 / ss) / numpy.sqrt(2.0 * numpy.pi * ss)

        return log(numpy.multiply.accumulate(single_probs)[-1])

    def _validate(self, param, value):

        if param != 'sigma':
            raise ParameterNotFoundError('GaussianErrorModel allows only for ' + 
                                         'one parameter called \'sigma\'')

        value = float(value.value)
        if value <= 0.0:
            raise ValueError('sigma has to be >= 0')

    def gradient(self, mock_data):
        """
        Currently hacked and tailored for the example in sandbox_simeon.py
        """
        
        return (mock_data[1] - mock_data[2]) / (self['sigma'].value ** 2)
