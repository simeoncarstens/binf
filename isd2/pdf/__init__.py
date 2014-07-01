"""
This module should provide general functionality and classes
related to probability density functions and their parameters.
"""
## TODO: Some of this might end up in CSB at some point.

from abc import ABCMeta, abstractmethod

from isd2 import AbstractISDNamedCallable

from csb.numeric import exp
from csb.statistics.pdf.parameterized import ParameterizedDensity


class ParameterNotFoundError(AttributeError):

    pass


class AbstractISDPDF(ParameterizedDensity, AbstractISDNamedCallable):

    __metaclass__ = ABCMeta

    def __init__(self, name='', **args):

        ParameterizedDensity.__init__(self)
        AbstractISDNamedCallable.__init__(self, name)

        self._var_param_types = {}

    def _set_original_variables(self):

        self._original_variables = self.variables.copy()

    @property
    def estimator(self):
        raise NotImplementedError
    @estimator.setter
    def estimator(self, strategy):
        pass

    @property
    def var_param_types(self):
        '''
        Empty by default, but for a conditional PDF to be built,
        one has to add AbstractParameter subclasses suiting the variables.
        '''
        return self._var_param_types.copy()
    def update_var_param_types(self, **values):
        self._var_param_types.update(**values)
    
    def estimate(self, data):
        raise NotImplementedError

    def conditional_factory(self, **fixed_vars):

        result = self.clone()
        result.fix_variables(**fixed_vars)

        return result            
    
    @abstractmethod
    def _evaluate_log_prob(self, **variables):
        pass

    def _evaluate(self, **variables):

        return exp(self.log_prob(**variables))

    def log_prob(self, **variables):

        vs = self._complete_variables(**variables)
        ## _complete_variables originally was supposed to update the variables
        ## dict in-place, but that didn't work and return it. Now here the dict
        ## still wasn't updated. Don't understand why...

        result = self._evaluate_log_prob(**vs)
        variables = self._reduce_variables(**vs)

        return result
    
    def gradient(self, **variables):

        vs = self._complete_variables(**variables)
        result = self._evaluate_gradient(**vs)
        variables = self._reduce_variables(**vs)

        return result

    def clone(self):

        pass

    def fix_variables(self, **fixed_vars):

        for v in fixed_vars:
            if v in self.variables:
                self._delete_variable(v)
                self._register(v)
                if v in self.var_param_types:
                    self[v] = self.var_param_types[v](fixed_vars[v], v)
                else:
                    raise ValueError('Parameter type for variable "'+v+'" not defined')

    def set_fixed_variables_from_pdf(self, pdf):

        self.fix_variables(**{p: pdf[p].value for p in pdf.parameters if not p in self.parameters})

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

