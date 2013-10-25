"""
This module contains all likelihoods occuring in ISD.
Technically, isd2.pdf is the wrong place for this module,
because a likelihood isn't a pdf (and shouldn't be derived from one).
"""

from isd2.pdf import AbstractISDPDF


class AbstractISDLikelihood(AbstractISDPDF):
    """
    TODO: In fact, the likelihood isn't a PDF, which means it shouldn't be derived from one
    """

    def __init__(self, forward_model, error_model):

        super(AbstractISDLikelihood, self).__init__()

        self._forward_model = forward_model
        self._error_model = error_model

    @property
    def forward_model(self):
        return self._forward_model

    @property
    def error_model(self):
        return self._error_model

    def log_prob(self, variables):

        return self.error_model.log_prob(self.forward_model(variables['structure']))

    def gradient(self, structure):

        pass

    def _register_var(self, var):

                self.variables.append(var)
