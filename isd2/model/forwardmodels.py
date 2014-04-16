"""
This module contains all forward models.
"""

from abc import abstractmethod, ABCMeta

from isd2.model import AbstractModel


class AbstractForwardModel(AbstractModel):

    '''
    The data argument is nonsense. Forward models
    don't know about data. But for now, I'm misusing
    it for various things.
    '''

    __meta__ = ABCMeta

    def __init__(self, name, parameters=[]):

        super(AbstractForwardModel, self).__init__(name, parameters)

    @abstractmethod
    def __call__(self, structure):
        pass

    @property
    def data(self):
        return self._data


class AbstractDifferentiableForwardModel(AbstractForwardModel):

    @abstractmethod
    def jacobi_matrix(self, **variables):
        pass
