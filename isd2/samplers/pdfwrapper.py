import numpy

from csb.statistics.pdf import AbstractDensity
from isd2.samplers import ISDState

class PDFWrapper(AbstractDensity):

    def __init__(self, isd2pdf):

        self.isd2pdf = isd2pdf
        self.vns = list(self.isd2pdf.variables)

    def log_prob(self, x):

        if isinstance(x, ISDState):
            return self.isd2pdf.log_prob(**{vn: x.variables[vn]
                                            for vn in self.vns})
        else:
            if len(self.vns) == 1:
                return self.isd2pdf.log_prob(**{self.vns[0]: x})
            else:
                raise('Ambiguous variables')

    def gradient(self, x, t=0.0):

        if isinstance(x, ISDState):
            return self.isd2pdf.gradient(**{vn: x.variables[vn]
                                            for vn in self.vns})
        else:
            if len(self.vns) == 1:
                return self.isd2pdf.gradient(**{self.vns[0]: x})
            else:
                raise('Ambiguous variables')
            
    @property
    def variables(self):
        return self.isd2pdf.variables
