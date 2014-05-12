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

        self._setup_parameters()
        self._components = dict(**self.priors)
        self._components.update(**self.likelihoods)
        self._register_component_variables(*self._get_component_variables())

        self._set_original_variables()

    def _setup_parameters(self):

        for L in self.likelihoods.values():
            for p in L.parameters:
                if p not in self.parameters:
                    self._register(p)
                    self[p] = L[p].__class__(L[p].value, 
                                             L[p].name,
                                             self[p])
                    # L[p].bind_to(self[p])

        for P in self.priors.values():
            for p in P.parameters:
                if p not in self.parameters:
                    self._register(p)
                    self[p] = P[p].__class__(P[p].value, 
                                             P[p].name, 
                                             self[p])
                    # P[p].bind_to(self[p])
    
    def _get_component_variables(self):

        vars = []
        diff_vars = []

        for c in self._components:
            for v in self._components[c].variables:
                vars.append(v)
                if v in self._components[c].differentiable_variables:
                    diff_vars.append(v)

        return set(vars), set(diff_vars)

    def _register_component_variables(self, vars, diff_vars):

        for var in vars:
            self._register_variable(str(var), differentiable=var in diff_vars)

    def _update_likelihood_parameters(self, params):

        for l in self._likelihoods.values():
            for p in l.parameters:
                ## likelihoods may have parameters which should not be sampled
                if p in params:
                    l[p].set(params[p])
            # l.set_params(**{x: params[x] for x in params if x in l.parameters})

    def _update_prior_parameters(self, params):
        
        for p in self._priors.values():
            for pm in p.parameters:
                ## priors may have parameters which should not be sampled
                if pm in params:
                    p[pm].set(params[pm])
            # p.set_params(**{x: params[x] for x in params if x in p.parameters})

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
            single_result = c.log_prob(**{v: mps[v] for v in comp_vars[c]})
            results.append(single_result)

        return results

    # def conditional_factory(self, **fixed_vars):

    #     ## TODO: implement clone methods
    #     from copy import deepcopy

    #     result = deepcopy(self)
        
    #     for n, p in self.priors.iteritems():
    #         common_vars = {v: fixed_vars[v] for v in p.variables 
    #                        if v in set(fixed_vars).intersection(set(p.variables))}
    #         if len(common_vars) > 0:
    #             result._priors[p.name] = p.conditional_factory(**common_vars)

    #     for n, l in self.likelihoods.iteritems():
    #         common_vars = {v: fixed_vars[v] for v in l.variables
    #                        if v in set(fixed_vars).intersection(set(l.variables))}
    #         if len(common_vars) > 0:
    #             result._likelihoods[l.name] = l.conditional_factory(**common_vars)

    #     result._setup_components_dict()
        
    #     for v in fixed_vars:
    #         result._delete_variable(v)
    #         result._register(v)
    #         if self.var_param_types[v]:
    #             result[v] = self.var_param_types[v](fixed_vars[v], v)
    #         else:
    #             raise('Parameter type for variable "'+v+'" not defined')

    #     result.log_prob = lambda **variables: result._eval_log_prob(**dict(variables, **fixed_vars))
    #     result.gradient = lambda **variables: result._eval_gradient(**dict(variables, **fixed_vars))

    #     return result

    def _evaluate_log_prob(self, **model_parameters):

        single_results = self._evaluate_components(**model_parameters)

        return numpy.sum(single_results)

    @property
    def likelihoods(self):
        return self._likelihoods

    @property
    def priors(self):
        return self._priors

    def _evaluate_gradients(self, **variables):

        vars = variables
        single_gradients = []
        
        for n, f in self._components.iteritems():
            if len(f.variables) > 0 and len(f.differentiable_variables) > 0:
                single_result = f.gradient(**{x: vars[x] for x in vars 
                                              if x in f.variables})

                single_gradients.append(single_result)

        return single_gradients

    def _evaluate_gradient(self, **model_parameters):

        self._update_nuisance_parameters(model_parameters)
        grad_evals = self._evaluate_gradients(**model_parameters)

        return numpy.sum(grad_evals, 0)

    def clone(self):

        copy = self.__class__({L: self.likelihoods[L].clone() for L in self.likelihoods},
                              {P: self.priors[P].clone() for P in self.priors}, 
                              self.name)

        for p in self.parameters:
            if not p in copy.parameters:
                copy._register(p)
                copy[p] = self[p].__class__(self[p].value, p)

        return copy
