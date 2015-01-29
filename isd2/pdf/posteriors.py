"""
This module contains all (conditional) posterior distributions occuring in ISD.
"""

from abc import ABCMeta, abstractmethod

import numpy

from csb.numeric import log, exp

from isd2 import AbstractISDNamedCallable
from isd2.core.composed import _get_component_var_param_types, _setup_variables, _setup_parameters, fix_variables
from isd2.pdf import AbstractISDPDF


class Posterior(AbstractISDPDF):

    def __init__(self, likelihoods, priors, name='the one and only posterior'):

        super(Posterior, self).__init__(name)

        self._likelihoods = likelihoods
        self._priors = priors

        self._components = dict(**self.priors)
        self._components.update(**self.likelihoods)
        
        self._setup_variables()
        self.update_var_param_types(**self._get_component_var_param_types())
        self._setup_parameters()

        self._set_original_variables()

    _get_component_var_param_types = _get_component_var_param_types

    _setup_variables = _setup_variables

    _setup_parameters = _setup_parameters

    fix_variables = fix_variables
    
    def _get_component_variables_list(self):

        return {c: c.variables for c in self._components.values()}

    def _evaluate_components(self, **model_parameters):
        
        mps = model_parameters
        results = []

        comp_vars = self._get_component_variables_list()
        
        for c in comp_vars:
            single_result = c.log_prob(**{v: mps[v] for v in comp_vars[c]})
            results.append(single_result)

        return results

    def _evaluate_log_prob(self, **model_parameters):

        single_results = self._evaluate_components(**model_parameters)

        return numpy.sum(single_results)

    @property
    def likelihoods(self):
        return self._likelihoods

    @property
    def priors(self):
        return self._priors

    def _evaluate_gradient(self, **variables):

        res = numpy.zeros(sum([len(variables[v]) for v in variables 
                               if v in self.differentiable_variables]))

        for n, f in self._components.iteritems():
            if len(f.variables) > 0 and len(f.differentiable_variables) > 0:
                res += f.gradient(**{x: variables[x] for x in variables 
                                     if x in f.variables})

        return res
    
    def clone(self):

        copy = self.__class__({L: self.likelihoods[L].clone() for L in self.likelihoods},
                              {P: self.priors[P].clone() for P in self.priors}, 
                              self.name)

        return copy

