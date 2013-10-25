"""
This module contains all forward models.
"""

from abc import abstractmethod

from isd2.model import AbstractModel


class AbstractForwardModel(AbstractModel):

    def __init__(self, name, data):

        super(AbstractForwardModel, self).__init__(name)

        self._data = data

    def __call__(self, structure):

        pass

    @property
    def data(self):
        return self._data


class AbstractDifferentiableForwardModel(AbstractForwardModel):

    @abstractmethod
    def jacobi_matrix(self, **variables):
        pass
