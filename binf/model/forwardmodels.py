"""
This module contains interfaces for forward models.
"""

from abc import abstractmethod, ABCMeta

from isd2.model import AbstractModel


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

    def _set_parameters(self, copy):
        
        for p in self.parameters:
            if not p in copy.parameters:
                copy._register(p)
                copy[p] = self[p].__class__(self[p].value, p)
                if p in copy.variables:
                    copy._delete_variable(p)
