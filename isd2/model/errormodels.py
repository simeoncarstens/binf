"""
This module contains all error models occuring in ISD.
"""

import numpy

from abc import ABCMeta, abstractmethod

from isd2.model import AbstractModel
from isd2.pdf import ParameterNotFoundError, AbstractISDPDF

from csb.numeric import log, exp


class AbstractErrorModel(AbstractISDPDF):

    __metaclass__ = ABCMeta
    
