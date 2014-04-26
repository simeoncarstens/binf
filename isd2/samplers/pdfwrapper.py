import numpy

from csb.statistics.pdf import AbstractDensity


class PDFWrapper(AbstractDensity):

    def __init__(self, isd2pdf):

        self.isd2pdf = isd2pdf
        self.vn = list(self.isd2pdf.variables)[0]

    def log_prob(self, x):

        return self.isd2pdf.log_prob(**{self.vn: x})

    def gradient(self, x, t=0.0):

        return self.isd2pdf.gradient(**{self.vn: x})

    @property
    def variables(self):
        return self.isd2pdf.variables
