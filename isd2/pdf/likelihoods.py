"""
This module contains all likelihoods occuring in ISD.
"""

import numpy

from abc import ABCMeta, abstractmethod

from csb.numeric import exp

from isd2.pdf import AbstractISDPDF

from scipy.sparse import coo_matrix


class AbstractLikelihood(AbstractISDPDF):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name, forward_model, error_model, data):

        super(AbstractLikelihood, self).__init__(name)
        
        self._forward_model = forward_model
        self._error_model = error_model
        self._data = data

        self._setup_parameters()
        
        self._set_original_variables()

    def _setup_parameters(self):

        for p in self._forward_model.get_params():
            self._register(p.name)
            self[p.name] = p.__class__(p.value, p.name, self[p.name])

        for p in self._error_model.get_params():
            self._register(p.name)
            self[p.name] = p.__class__(p.value, p.name, self[p.name])

    @property
    def forward_model(self):
        return self._forward_model

    @property
    def error_model(self):
        return self._error_model

    @property
    def data(self):
        return self._data

    def _evaluate_log_prob(self, **variables):

        fwm_variables = {v: variables[v] for v in variables if v in self.forward_model.variables}
        em_variables = {v: variables[v] for v in variables if v in self.error_model.variables}
        mock_data = self.forward_model(**fwm_variables)
        
        return self.error_model.log_prob(mock_data=mock_data, **em_variables)

    def _evaluate_gradient(self, **variables):

        fwm_variables = {v: variables[v] for v in variables if v in self.forward_model.variables}
        em_variables = {v: variables[v] for v in variables if v in self.error_model.variables}
        mock_data = self.forward_model(**fwm_variables)
        dfm = self.forward_model.jacobi_matrix(**fwm_variables)
        emgrad = self.error_model.gradient(mock_data=mock_data, **em_variables)

        return dfm.dot(emgrad)

    def clone(self):

        copy = self.__class__(self.name,
                              self.forward_model.clone(),
                              self.error_model.clone(),
                              self.data)

        copy.set_fixed_variables_from_pdf(self)
        
        return copy

    def conditional_factory(self, **fixed_vars):

        result = self.clone()

        result.fix_variables(**result._get_variables_intersection(fixed_vars))
        
        return result

