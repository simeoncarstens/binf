"""
This module contains interfaces for forward models.
"""

from abc import abstractmethod, ABCMeta

from binf.model import AbstractModel


class AbstractForwardModel(AbstractModel):

    __meta__ = ABCMeta

    @abstractmethod
    def __init__(self, name, parameters=[]):

        super(AbstractForwardModel, self).__init__(name, parameters)

    @property
    def data(self):
        return self._data

    def jacobi_matrix(self, **variables):

        self._complete_variables(variables)
        result = self._evaluate_jacobi_matrix(**variables)

        return result

    @abstractmethod
    def _evaluate_jacobi_matrix(self, **model_parameters):

        self._check_differentiability(**model_parameters)

    @abstractmethod
    def clone(self):

        pass

    def fix_variables(self, **fixed_vars):
        """
        Sets ('fixes') specific variables to values given as keyword
        arguments by removing these variables from the list of registered
        variables and registering corresponding parameters to this object.
        """
        for v in fixed_vars:
            if v in self.variables:
                self._delete_variable(v)
                self._register(v)
                if v in self.var_param_types:
                    self[v] = self.var_param_types[v](fixed_vars[v], v)
                else:
                    msg = 'Parameter type for variable "{}" not defined'.format(v)
                    raise ValueError(msg)
            else:
                msg = '{} is not a variable of {}'.format(self.__repr__(), v)
                raise ValueError(msg)
    
    def _set_parameters(self, copy):
        
        for p in self.parameters:
            if not p in copy.parameters:
                copy._register(p)
                copy[p] = self[p].__class__(self[p].value, p)
                if p in copy.variables:
                    copy._delete_variable(p)
