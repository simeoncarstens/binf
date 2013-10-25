"""
"""
## TODO: move test code somewhere else, complete documentation

import numpy

from abc import abstractmethod

from csb.numeric import log
from csb.statistics.pdf import AbstractDensity, ParameterNotFoundError
from csb.statistics.pdf.parameterized import Parameter, ParameterizedDensity

from numpy import sqrt, pi

class InvalidOperationError(Exception):
    pass
    
class Location(Parameter):
    
    def _validate(self, value):
        return float(value)

class Scale(Parameter):
    
    def _validate(self, value):
        return float(value)
    
    def _compute(self, base_value):
        
        if base_value == 0.0:
            return numpy.inf
        else:
            return 1.0 / base_value ** 0.5
        
    def bind_to(self, base):
        
        if base.name != "precision":
            raise ValueError(base)
                
        super(Scale, self).bind_to(base)
    
class Precision(Parameter):
    
    def _validate(self, value):
        
        if value < 0:
            raise ValueError(value)
        return float(value)
    
                
class FancyGaussian(ParameterizedDensity):        
        
    def __init__(self, mu=0, precision=1):
        
        super(FancyGaussian, self).__init__()
        
        self._register('mu')
        self._register('sigma')
        self._register('precision')
        
        loc = Location(mu)
        prec = Precision(precision) 
        sigma = Scale(0)
        sigma.bind_to(prec)
        
        self.set_params(mu=loc, sigma=sigma, precision=prec)

    def __setitem__(self, param, value):

        if param in self._params: 

            self._validate(param, value)
            self._params[param] = value
        else:
            raise ParameterNotFoundError(param)
        
    @property
    def mu(self):
        return self['mu'].value
    @mu.setter
    def mu(self, value):
        self['mu'] = float(value)
        
    @property
    def sigma(self):
        return self['sigma'].value
    
    @property
    def precision(self):
        return self['precision'].value  
    @precision.setter
    def precision(self, value):
        self['precision'].set(value)
            
    def log_prob(self, x):

        mu = self.mu
        sigma = self.sigma
        
        return log(1.0 / sqrt(2 * pi * sigma ** 2)) - (x - mu) ** 2 / (2 * sigma ** 2)

if __name__ == "__main__":

    pdf = FancyGaussian(1, 2)
    print pdf.mu, pdf.sigma, pdf.precision
        
    pdf = FancyGaussian(2, 5)
    print pdf.mu, pdf.sigma, pdf.precision
    
    pdf['precision'].set(2)
    print pdf.mu, pdf.sigma, pdf.precision
    
    try:
        pdf['sigma'].set(2)
    except InvalidOperationError as ex:
        print str(ex)
        
        
        
    
        
