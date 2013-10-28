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
        self._components = dict(self._priors, **self._likelihoods)

        self._register_component_variables(self._get_component_variables())

    def _get_component_variables(self):

        vars = []

        for c in self._components:
            for v in self._components[c].variables:
                vars.append(v)

        return set(vars)

    def _register_component_variables(self, vars):

        for var in vars:
            self._register_variable(str(var)) 

    def _update_likelihood_parameters(self, params):

        for l in self._likelihoods.values():
            l.set_params(**{x: params[x] for x in params if x in l.parameters})

    def _update_prior_parameters(self, params):
        
        for p in self._priors.values():
            p.set_params(**{x: params[x] for x in params if x in p.parameters})

    def _update_nuisance_parameters(self, params):

        self._update_likelihood_parameters(params)
        self._update_prior_parameters(params)

    def _get_component_variables_list(self):

        return {c: c.variables for c in self._components.values()}

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

    def conditional_factory(self, **fixed_vars):

        ## TODO: implement clone methods
        from copy import deepcopy

        result = deepcopy(self)
        
        for n, p in self.priors.iteritems():
            common_vars = {v: fixed_vars[v] for v in p.variables 
                           if v in set(fixed_vars).intersection(set(p.variables))}
            if len(common_vars) > 0:
                result._priors[p.name] = p.conditional_factory(**common_vars)

        for n, l in self.likelihoods.iteritems():
            common_vars = {v: fixed_vars[v] for v in l.variables
                           if v in set(fixed_vars).intersection(set(l.variables))}
            if len(common_vars) > 0:
                result._likelihoods[l.name] = l.conditional_factory(**common_vars)

        for v in fixed_vars:
            result._delete_variable(v)
            result._register(v)
            result[v] = fixed_vars[v]

        result.log_prob = lambda **variables: result._eval_log_prob(**dict(variables, **fixed_vars))

        return result

    def _eval_log_prob(self, **model_parameters):
        """
        This is really bad. But so far it is needed to make the conditional_factory() work
        """

        single_results = self._evaluate_components(**model_parameters)

        return log(numpy.multiply.accumulate(single_results)[-1])
        
    
    def log_prob(self, **model_parameters):
        """
        See _eval_log_prob()
        """

        return self._eval_log_prob(**model_parameters)

        # single_results = self._evaluate_components(**model_parameters)

        # return log(numpy.multiply.accumulate(single_results)[-1])

    @property
    def likelihoods(self):
        return self._likelihoods

    @property
    def priors(self):
        return self._priors


class DifferentiablePosterior(Posterior, AbstractISDNamedCallable):

    def conditional_factory(self, **fixed_vars):

        result = super(DifferentiablePosterior, self).conditional_factory(**fixed_vars)
        result.gradient = lambda **variables: self.gradient(**dict(variables, **fixed_vars))

        return result

    def _evaluate_gradients(self, **variables):

        vars = variables
        single_gradients = []
        
        for n, f in self._components.iteritems():
            if len(f.variables) > 0:
                single_result = f.gradient(**{x: vars[x] for x in vars 
                                              if x in f.variables})

                single_gradients.append(single_result)

        return single_gradients

    def gradient(self, **model_parameters):

        self._update_nuisance_parameters(model_parameters)

        grad_evals = self._evaluate_gradients(**model_parameters)

        return numpy.sum(grad_evals, 0)
