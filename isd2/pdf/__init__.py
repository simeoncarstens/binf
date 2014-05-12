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

    # def conditional_factory(self, **fixed_vars):

    #     result = self.clone()

    #     ## Does this break encapsulation?
        
    #     for v in fixed_vars:
    #         result._delete_variable(v)
    #         result._register(v)
    #         if self.var_param_types[v]:
    #             result[v] = self.var_param_types[v](fixed_vars[v], v)
    #         else:
    #             raise('Parameter type for variable "'+v+'" not defined')

    #     result.log_prob = lambda **variables: self.log_prob(**dict(variables, **fixed_vars))
    #     result.__call__ = lambda **variables: self.__call__(**dict(variables, **fixed_vars))

    #     return result

    def conditional_factory(self, **fixed_vars):

        result = self.clone()

        for v in fixed_vars:
            result._delete_variable(v)
            result._register(v)
            if self.var_param_types[v]:
                result[v] = self.var_param_types[v](fixed_vars[v], v)
            else:
                raise('Parameter type for variable "'+v+'" not defined')

        return result            
    
    @abstractmethod
    def _evaluate_log_prob(self, **variables):

        pass

    def log_prob(self, **variables):

        vs = self._complete_variables(**variables)
        ## _complete_variables originally was supposed to update the variables
        ## dict in-place, but that didn't work and return it. Now here the dict
        ##  still wasn't updated. Don't understand why...
        result = self._evaluate_log_prob(**vs)
        variables = self._reduce_variables(**vs)

        return result

    def gradient(self, **variables):

        print self, variables.keys()
        vs = self._complete_variables(**variables)
        result = self._evaluate_gradient(**vs)
        variables = self._reduce_variables(**vs)

        return result

    def __call__(self, **variables):

        return exp(self.log_prob(**variables))

    def clone(self):

        pass      
