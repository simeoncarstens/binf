"""
"""
import numpy

from abc import abstractmethod

from csb.numeric import log
from csb.statistics.pdf import AbstractDensity, ParameterNotFoundError

from numpy import sqrt, pi

class InvalidOperationError(Exception):
    pass
    
class Parameter(object):
    """
    Abstract parameterization, which can exist independently or be coupled 
    to other parameters upon request. Virtual/coupled/derived parameters cannot
    be overwritten explicitly, but their values will get recomputed once their
    corresponding base parameters get updated. This is a lazy process -- parameter
    recalculation happens only when an out of date parameter is requested. 
    """
    
    NULL = None
    
    def __init__(self, value=NULL, name=NULL):
        
        self._derivatives = set()
        self._base = None
        self._consistent = False
        
        if name is Parameter.NULL:
            name = self.__class__.__name__.lower()
        
        self._name = str(name)
        self._value = Parameter.NULL
        
        self._update(value)
    
    @property
    def name(self):
        """
        Parameter name
        """
        return self._name
    
    @property
    def value(self):
        """
        Parameter value (guaranteed to be up to date)
        """
        self._ensure_consistency()
        return self._value
    
    @property
    def is_virtual(self):
        """
        True if this parameter is virtual (computed)
        """
        return self._base is not None

    def set(self, value):
        """
        Update the value of this parameter. This is not possible for 
        virtual parameters.
        """
        
        if self.is_virtual:
            raise InvalidOperationError(
                    "Virtual parameters can't be updated explicitly")
            
        self._value = value
        self._invalidate()

    def bind_to(self, parameter):
        """
        Bind the current parameter to a base parameter. This converts 
        the current parameter to a virtual one, whose value will get 
        implicitly updated to be consistent with its base.
        """

        if not isinstance(parameter, Parameter):
            raise TypeError(parameter)        
        
        if self.is_virtual:
            msg = "Parameter {0.name} is already bound to {1.name}"
            raise InvalidOperationError(msg.format(self, self._base))
        
        self._set_base(parameter)
        self._base._add_derived(self)
        
        self._invalidate()  
        
    def _set_base(self, parameter):
        self._base = parameter
        
    def _add_derived(self, parameter):
        self._derivatives.add(parameter) 
        
    def _invalidate(self):
        """
        Mark self and its virtual children as inconsistent
        """

        for p in self._derivatives: 
            p._invalidate()
            
        self._consistent = False            
            
    def _update(self, value):
        """
        Overwrite the current value of the parameter. This triggers
        an abstract (custom) validation hook, but has no side effects 
        (i.e. it doesn't propagate!)
        """
        sanitized = self._validate(value)
        self._value = sanitized
        
    @abstractmethod
    def _validate(self, value):
        """
        Validate and sanitize the specified value before assignment.
        @return: sanitized value
        
        @raise ParameterValueError: on invalid value
        """
        return value
    
    def _compute(self, base_value):
        """
        Compute a new value for the current parameter given the value
        of a base parameter (assuming self.is_virtual). By default this raises
        NotImplementedError (i.e. self is not a virtual parameter).
        """
        raise NotImplementedError()
            
    def _ensure_consistency(self, force=False):
        """
        Make sure that the current value is up to date. If it isn't,
        trigger a real-time cascaded update from the non-virtual root 
        to all virtual nodes. Also mark all nodes consistent in the course of
        doing this update. 
        """        
        if self._consistent and not force:
            return
        
        root = self.find_base_parameter()
        root._recompute_derivatives()
        
    def _recompute_derivatives(self):
        """
        Recompute all derived parameters starting from self and mark 
        them consistent.
        """
        
        if self.is_virtual:
            value = self._compute(self._base._value)
            self._update(value)
        
        self._consistent = True
        
        for p in self._derivatives:
            p._recompute_derivatives()        
        
    def find_base_parameter(self):
        """
        Find and return the non-virtual base parameter that is the root
        of the current hierarchy. If self is not virtual, return self.
        """
        
        root = self
        while root.is_virtual:
            root = root._base
            
        return root 
    

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
    
                
class FancyGaussian(AbstractDensity):        
        
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
        
        
        
    
        
