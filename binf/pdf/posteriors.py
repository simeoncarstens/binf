"""
This module contains implementations of posterior distribution interfaces
"""

from abc import ABCMeta, abstractmethod

import numpy

from csb.numeric import log, exp

from isd2 import AbstractISDNamedCallable
from isd2.pdf import AbstractISDPDF


class Posterior(AbstractISDPDF):

    def __init__(self, likelihoods, priors, name='the one and only posterior'):
        """
        A PDF object implementing a posterior distribution consisting of
        multiple likelihoods and priors

        :param likelihoods: name / object pairs of likelihoods modeling
                            different data-generating processes
        :type likelihoods: dict

        :param priors: name / object pairs of priors  modeling prior
                       information about the modeling parameters

        :param name: some name for this object
        :type name: str
        """
        super(Posterior, self).__init__(name)

        self._likelihoods = likelihoods
        self._priors = priors

        self._setup_parameters()
        self._components = dict(**self.priors)
        self._components.update(**self.likelihoods)
        self._register_component_variables(*self._get_component_variables())
        
        self._set_original_variables()

    def _setup_parameters(self):
        """
        Sets up ('inherits') parameters from likelihoods and priors
        """
        for comps in (self.likelihoods, self.priors):
            for comp in comps.values():
                for p in comp.parameters:
                    if p not in self.parameters:
                        self._register(p)
                        self[p] = comp[p].__class__(comp[p].value, 
                                                    comp[p].name)
                    comp[p].bind_to(self[p])

    def _get_component_variables(self):
        """
        Retrieves variables from likelihoods and priors

        :returns: sets of variables, fixed variables, variables the
                  posterior distribution can be differentiated w.r.t.
                  and the variable parameter types
        :rtype: (set, set, set, dict)
        """
        vars = []
        fixed_vars = []
        diff_vars = []
        var_param_types = []

        for c in self._components:
            for v in self._components[c].variables:
                vars.append(v)
                var_param_types.append(self._components[c].var_param_types[v])
                if v in self._components[c].differentiable_variables:
                    diff_vars.append(v)
            for p in self._components[c].parameters:
                if p in self._components[c]._original_variables:
                    fixed_vars.append(p)
                    ## Might or might not work, atm I'm not clear about the status 
                    ## of the differentiable_variable thing:
                    # if p in self._components[c].differentiable_variables:
                    #     diff_vars.append(v) 

        return set(vars), set(fixed_vars), set(diff_vars), var_param_types

    def _register_component_variables(self, vars, fixed_vars, diff_vars,
                                      var_param_types):
        """
        Registers variables of likelihoods and priors as variables of
        this object and updates the variable parameter type dictionary

        :param vars: set of variables
        :type vars: set

        :param fixed_vars: set of fixed variables
        :type fixed_vars: set

        :param diff_vars: set of variables the posterior distribution can
                          be differentiated w.r.t.
        :type diff_vars: set

        :param var_param_types: the parameter type for each variable
        :type var_param_types: dict
        """

        for var in vars:
            self._register_variable(str(var), differentiable=var in diff_vars)

        for fixed_var in fixed_vars:
            self._original_variables.update({fixed_var})

        self.update_var_param_types(**{var: _type for var, _type
                                       in zip(vars, var_param_types)})

    def _get_component_variables_list(self):
        """
        For each component (likelihood or prior), returns its variables

        :returns: set of variables for each component
        :rtype: dict
        """
        return {c: c.variables for c in self._components.values()}

    def _evaluate_components(self, **model_parameters):
        r"""
        Evaluates the log-probabilities of all components

        :param \**model_parameters: parameters of the model. To be consistent,
                                    this should probably called variables
        :type \**model_parameters: dict
        
        :returns: list of log-probabilities
        :rtype: list
        """
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
        """
        Returns the likelihoods of this posterior distribution

        :returns: likelihoods
        :rtype: dict
        """
        return self._likelihoods

    @property
    def priors(self):
        """
        Returns the priors of this posterior distribution

        :returns: priors
        :rtype: dict
        """
        return self._priors

    def _evaluate_gradient(self, **variables):

        vars = variables

        res = numpy.zeros(sum([len(variables[v])
                               if hasattr(variables[v], '__len__') else 1 
                               for v in variables
                               if v in self.differentiable_variables]))

        for n, f in self._components.iteritems():
            if len(f.variables) > 0 and len(f.differentiable_variables) > 0:
                res += f.gradient(**{x: vars[x] for x in vars 
                                     if x in f.variables})

        return res
    
    def clone(self):

        copy = self.__class__({L: self.likelihoods[L].clone()
                               for L in self.likelihoods},
                              {P: self.priors[P].clone()
                               for P in self.priors}, 
                              self.name)

        copy.set_fixed_variables_from_pdf(self)
        
        return copy

    def conditional_factory(self, **fixed_vars):

        cond_likelihoods = {L: self.likelihoods[L].conditional_factory(**fixed_vars) 
                            for L in self.likelihoods}
        cond_priors = {P: self.priors[P].conditional_factory(**fixed_vars) 
                       for P in self.priors}

        copy = self.__class__(cond_likelihoods, cond_priors, self.name)
        
        return copy
        
