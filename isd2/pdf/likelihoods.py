"""
This module contains all likelihoods occuring in ISD.
"""

import numpy

from abc import ABCMeta, abstractmethod

from csb.numeric import exp

from isd2.pdf import AbstractISDPDF

class AbstractLikelihood(AbstractISDPDF):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name, forward_model, error_model, data):

        super(AbstractLikelihood, self).__init__(name)
        
        self._forward_model = forward_model
        self._error_model = error_model
        self._data = data

        fwm_params = {p.name: p for p in self._forward_model.get_params()}
        em_params = {p.name: p for p in self._error_model.get_params()}
        self._nuisance_params = dict(fwm_params, **em_params)

    #     self._setup_parameters()

    # def _setup_parameters(self):

    #     for p in self._forward_model.get_params():
    #         self._register(p.name)
    #         self[p.name] = p.__class__(p.value, p.name)
    #         p.bind_to(self[p.name])

    #     for p in self._error_model.get_params():
    #         self._register(p.name)
    #         self[p.name] = p.__class__(p.value, p.name)
    #         p.bind_to(self[p.name])

    @property
    def forward_model(self):
        return self._forward_model

    @property
    def error_model(self):
        return self._error_model

    @property
    def data(self):
        return self._data

    def _split_variables(self, **variables):

        return {v: variables[v] for v in variables if not v in self._nuisance_params},\
               {v: variables[v] for v in variables if v in self._nuisance_params}

    def _update_nuisance_parameters(self, **nuisance_params):

        for p in nuisance_params:
            self._nuisance_params[p].set(nuisance_params[p])

    def log_prob(self, **variables):

        vois, nps = self._split_variables(**variables)
        self._update_nuisance_parameters(**nps)
        
        return self.error_model.log_prob(self.forward_model(**vois))

    def gradient(self, **variables):

        vois, nps = self._split_variables(**variables)
        self._update_nuisance_parameters(**nps)
        
        l = self(**variables)
        mock_data = self.forward_model(**vois)
        dfm = self.forward_model.jacobi_matrix(**vois)
        emgrad = self.error_model.gradient(mock_data)
        
        return numpy.dot(dfm, emgrad)
