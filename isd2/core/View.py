"""
Implements a View on objects.

The View provides a (passive) representation of an object it has
been attached to.
"""
from isd2 import DEBUG
from isd2.core.Viewable import Viewable
from abc import ABCMeta, abstractmethod

class View(object):
    """
    Abstract Observer class that can be attached to viewable objects.
    If the status of the object changes the view is invalidated.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, viewable):
        """
        Initialized with a viewable object.
        """
        if not isinstance(viewable, Viewable):
            msg = '{0} can only be initialized with a viewable object'.format(self)
            raise TypeError(msg)

        self.invalidate()
        viewable.add_view(self)
        self._object = viewable
        
        ## TODO: is there a better name?
        self.__values = None

    @property
    def _values(self):
        if self.__values is None:
            raise ValueError('values not set')
        return self.__values

    @_values.setter
    def _values(self, values):
        self.__values = values
    
    def __del__(self):
        self._object.del_view(self)

    def invalidate(self):
        if DEBUG: print '{0}.invalidate'.format(self.__class__.__name__)
        self._valid = False

    def is_valid(self):
        return self._valid

    @abstractmethod
    def update(self):
        """
        This method must set __values to an object that is different from None 
        """
        pass

    def get(self):
        if DEBUG: print '{0}.get'.format(self.__class__.__name__)

        if not self.is_valid():
            self.update()
            self._valid = True

        return self._values
    
