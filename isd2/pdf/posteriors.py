"""
This module contains all (conditional) posterior distributions occuring in ISD.
"""

from abc import ABCMeta, abstractmethod

import numpy

from csb.numeric import log, exp

from isd2 import AbstractISDNamedCallable
from isd2.pdf import AbstractISDPDF


class Posterior(AbstractISDPDF):

    def __init__(self, likelihoods, priors, name='the one and only posterior'):

        super(Posterior, self).__init__(name)

        self._likelihoods = likelihoods
        self._priors = priors
        self._components = self._priors + self._likelihoods

        self._register_component_variables(self._get_component_variables())

    def _get_component_variables(self):

        vars = [var for x in self._likelihoods + self._priors for var in x.variables]

        return set(vars)

    def _register_component_variables(self, vars):

        for var in vars:
            self._register_variable(str(var)) 

    def _update_error_model_parameters(self, params):

        for l in self._likelihoods:
            l.error_model.set_params(**{x: params[x] for x in params 
                                        if x in l.error_model.parameters})

    def _update_prior_parameters(self, params):

        for p in self._priors:
            p.set_params(**{x: params[x] for x in params 
                            if x in p.parameters})

    def _update_forward_model_parameters(self, params):

        for l in self._likelihoods:
            l.forward_model.set_params(**{x: params[x] for x in params 
                                          if x in l.forward_model.parameters})

    def _update_nuisance_parameters(self, params):

        self._update_forward_model_parameters(params)
        self._update_error_model_parameters(params)
        self._update_prior_parameters(params)

    def _get_component_variables_list(self):

        return {c: c.variables for c in self.priors + self.likelihoods}

    def _evaluate_components(self, **model_parameters):

        self._update_nuisance_parameters(model_parameters)
        
        mps = model_parameters
        results = []

        comp_vars = self._get_component_variables_list()

        for c in comp_vars:

            single_result = c(**{v: mps[v] for v in comp_vars[c]})
            results.append(single_result)

            if single_result < 1e-30:
                break

        return results

    def log_prob(self, **model_parameters):

        single_results = self._evaluate_components(**model_parameters)

        return log(numpy.multiply.accumulate(single_results)[-1])

    @property
    def likelihoods(self):
        return self._likelihoods

    @property
    def priors(self):
        return self._priors


class ConditionedPosterior(Posterior):

    def __init__(self, likelihoods, priors, fixed_parameters):

        super(ConditionedPosterior, self).__init__(likelihoods, priors)

        self._fixed_parameters = fixed_parameters
        self._delete_variables(self._fixed_parameters)
        self._setup_params(self._fixed_parameters)

    def _setup_params(self, fixed_parameters):

        sub_params = []

        for l in self.likelihoods:
            for p in l.forward_model.get_params():
                sub_params.append(p)
            for p in l.error_model.get_params():
                sub_params.append(p)
                
        for p in self.priors:
            for pprime in p.get_params():
                sub_params.append(pprime)

        unique_sub_params = {p.name: p for p in sub_params}
        
        params = {p: unique_sub_params[p].__class__(name=unique_sub_params[p].name)
                  for p in unique_sub_params if p in fixed_parameters}

        for p in params:
            unique_sub_params[p].bind_to(params[p])
            self._register(params[p].name)
            self[p] = params[p]

    def _delete_variables(self, fixed_parameters):
        
        for p in fixed_parameters:
            self._delete_variable(p)

    def log_prob(self, **variables):

        return super(ConditionedPosterior, self).log_prob(**variables)


class DifferentiableConditionedPosterior(ConditionedPosterior,
                                         AbstractISDNamedCallable):

    def _evaluate_gradients(self, **variables):

        vars = variables
        single_gradients = []
        
        for f in self._components:
            if self.variables != f.variables:
                continue
            single_result = f.gradient(**{x: vars[x] for x in vars if x in f.variables 
                                          and not x in self._fixed_parameters})
            single_gradients.append(single_result)

        return single_gradients

    def gradient(self, **model_parameters):

        self._update_nuisance_parameters(model_parameters)

        grad_evals = self._evaluate_gradients(**model_parameters)

        return numpy.sum(grad_evals, 0)
