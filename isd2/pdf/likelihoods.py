"""
This module contains all likelihoods occuring in ISD.
"""

import numpy

from csb.numeric import exp

from isd2.pdf import AbstractISDPDF


class Likelihood(AbstractISDPDF):

    def __init__(self, name, forward_model, error_model):

        super(Likelihood, self).__init__(name)
        
        self._forward_model = forward_model
        self._error_model = error_model

        self._inherit_variables()

        self._setup_parameters()
        
        self._set_original_variables()

    def _inherit_variables(self):

        fm = self._forward_model
        em = self._error_model
        
        for v in fm.variables:
            self._register_variable(v, differentiable=v in fm.differentiable_variables)
            self.update_var_param_types(**{v: fm.var_param_types[v]})

        for v in em.variables:
            if not v == 'mock_data':
                self._register_variable(v, differentiable=v in em.differentiable_variables)
                self.update_var_param_types(**{v: em.var_param_types[v]})

    def _setup_parameters(self):

        for p in self._forward_model.get_params():
            self._register(p.name)
            self[p.name] = p.__class__(p.value, p.name, self[p.name])
            p.bind_to(self[p.name])

        for p in self._error_model.get_params():
            self._register(p.name)
            self[p.name] = p.__class__(p.value, p.name, self[p.name])
            p.bind_to(self[p.name])

    def _setup_fixed_variable_parameters(self):

        for model in (self._forward_model, self._error_model):
            for p in model.get_params():
                var_param_type = model.var_param_types[p.name]
                model[p.name] = var_param_type(self[p.name].value, p.name)
                model[p.name].bind_to(self[p.name])

    @property
    def forward_model(self):
        return self._forward_model

    @property
    def error_model(self):
        return self._error_model

    def _split_variables(self, variables):

        fwm_variables = {v: variables[v] for v in variables if v in self.forward_model.variables}
        em_variables = {v: variables[v] for v in variables if v in self.error_model.variables}

        return fwm_variables, em_variables

    def _evaluate_log_prob(self, **variables):

        fwm_variables, em_variables = self._split_variables(variables)
        mock_data = self.forward_model(**fwm_variables)
        
        return self.error_model.log_prob(mock_data=mock_data, **em_variables)

    def _evaluate_gradient(self, **variables):

        fwm_variables, em_variables = self._split_variables(variables)
        mock_data = self.forward_model(**fwm_variables)
        dfm = self.forward_model.jacobi_matrix(**fwm_variables)
        emgrad = self.error_model.gradient(mock_data=mock_data, **em_variables)

        return dfm.dot(emgrad)

    def clone(self):

        copy = self.__class__(self.name,
                              self.forward_model.clone(),
                              self.error_model.clone(),
                              self.data)

        # copy.set_fixed_variables_from_pdf(self)
        
        return copy

    def fix_variables(self, **fixed_vars):

        super(Likelihood, self).fix_variables(**fixed_vars)

        # self._setup_fixed_variable_parameters()

    def conditional_factory(self, **fixed_vars):

        fwm = self.forward_model.clone()
        # fwm.fix_variables(**fwm._get_variables_intersection(**fixed_vars))
	fwm.fix_variables(**fwm._get_variables_intersection(fixed_vars))
        em = self.error_model.conditional_factory(**self.error_model._get_variables_intersection(fixed_vars))
        result = self.__class__(self.name, fwm, em)
        result.fix_variables(**result._get_variables_intersection(fixed_vars))

        return result            

