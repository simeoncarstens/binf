"""
This module contains all forward models.
"""

from abc import abstractmethod, ABCMeta

from isd2.model import AbstractModel


class AbstractForwardModel(AbstractModel):

    __meta__ = ABCMeta

    def __init__(self, name, parameters=[]):

        super(AbstractForwardModel, self).__init__(name, parameters)

    @abstractmethod
    def __call__(self, model_parameters):
        pass

    @property
    def data(self):
        return self._data

    @abstractmethod
    def jacobi_matrix(self, **model_parameters):

        self._check_differentiability(**model_parameters)
