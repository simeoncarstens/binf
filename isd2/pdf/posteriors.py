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
                                             L[p].name)
                    L[p].bind_to(self[p])

        for P in self.priors.values():
            for p in P.parameters:
                if p not in self.parameters:
                    self._register(p)
                    self[p] = P[p].__class__(P[p].value, 
                                             P[p].name)
                    P[p].bind_to(self[p])
    
    def _get_component_variables(self):

        vars = []
        diff_vars = []
        var_param_types = []

        for c in self._components:
            for v in self._components[c].variables:
                vars.append(v)
                var_param_types.append(self._components[c].var_param_types[v])
                if v in self._components[c].differentiable_variables:
                    diff_vars.append(v)

        return set(vars), set(diff_vars), var_param_types

    def _register_component_variables(self, vars, diff_vars, var_param_types):

        for var in vars:
            self._register_variable(str(var), differentiable=var in diff_vars)

        self.update_var_param_types(**{var: _type for var, _type in zip(vars, var_param_types)})

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

        vars = variables

        res = numpy.zeros(sum([len(variables[v]) for v in variables 
                               if v in self.differentiable_variables]))

        for n, f in self._components.iteritems():
            if len(f.variables) > 0 and len(f.differentiable_variables) > 0:
                res += f.gradient(**{x: vars[x] for x in vars 
                                     if x in f.variables})


        return res
    
    def clone(self):

        copy = self.__class__({L: self.likelihoods[L].clone() for L in self.likelihoods},
                              {P: self.priors[P].clone() for P in self.priors}, 
                              self.name)

        for p in self.parameters:
            if not p in copy.parameters:
                copy._register(p)
                copy[p] = self[p].__class__(self[p].value, p)

        return copy

    def conditional_factory(self, **fixed_vars):

        conditional_likelihoods = {L: self.likelihoods[L].conditional_factory(**fixed_vars) 
                                   for L in self.likelihoods}
        conditional_priors = {P: self.priors[P].conditional_factory(**fixed_vars) 
                              for P in self.priors}

        copy = self.__class__(conditional_likelihoods, conditional_priors, self.name)

        for v in copy.variables.difference(self.variables):
            copy._delete_variable(v)

        return copy
        
