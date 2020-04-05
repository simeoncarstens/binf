"""
This module contains interfaces for error models
"""

import numpy

from abc import ABCMeta, abstractmethod

from binf.model import AbstractModel
from binf.pdf import ParameterNotFoundError, AbstractBinfPDF

from csb.numeric import log, exp


class AbstractErrorModel(AbstractBinfPDF, metaclass=ABCMeta):

    def __init__(self, name, data):

        super(AbstractErrorModel, self).__init__(name)
        self.data = data
