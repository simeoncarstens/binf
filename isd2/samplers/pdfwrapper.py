import numpy

from csb.statistics.pdf import AbstractDensity


class PDFWrapper(AbstractDensity):

    def __init__(self, isd2pdf):

        self.isd2pdf = isd2pdf
        self.vns = list(self.isd2pdf.variables)

    def log_prob(self, x):

        return self.isd2pdf.log_prob(**{vn: x.variables[vn]
                                        for vn in self.vns})

    def gradient(self, x, t=0.0):

        return self.isd2pdf.gradient(**{vn: x.variables[vn]
                                        for vn in self.vns})

    @property
    def variables(self):
        return self.isd2pdf.variables
