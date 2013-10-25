"""
This module contains all prior distributions occuring in ISD.
All these priors are ultimately derived from isd2.pdf.AbstractISDPDF
"""

from abc import abstractmethod

from isd2.pdf import AbstractISDPDF


class AbstractPrior(AbstractISDPDF):

    pass


class AbstractDifferentiablePrior(AbstractPrior):

    @abstractmethod
    def gradient(self, **variables):

        pass
