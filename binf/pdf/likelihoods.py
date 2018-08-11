"""
This module contains interfaces for likelihood functions
"""

import numpy

from csb.numeric import exp

from binf.pdf import AbstractBinfPDF


class Likelihood(AbstractBinfPDF):

    def __init__(self, name, forward_model, error_model):
        """
        A likelihood function (not exactly a PDF...) which is composed
        of a forward model to back-calculate idealized data from
        model parameters and an error model modeling deviations of
        the data from the idealized data

        :param name: some name for this object
        :type name: str

        :param forward_model: forward model to back-calculate data from
        :type forward_model: :class:`.AbstractForwardModel`

        :param error_model: error model to model deviations of the data
                            from the back-calculated data
        :type error_model: :class:`.AbstractErrorModel`
        """
        super(Likelihood, self).__init__(name)
        
        self._forward_model = forward_model
        self._error_model = error_model

        self._inherit_variables()

        self._setup_parameters()
        
        self._set_original_variables()

    def _inherit_fwm_variables(self):
        """
        Retrieves variables from forward model and adds them
        to this objects's own set of variables
        """
        fwm = self._forward_model
        for v in fwm._original_variables:
            if v in fwm.parameters:
                self._original_variables.update({v})
            else:
                self._register_variable(v, differentiable=v in
                                        fwm.differentiable_variables)
            self.update_var_param_types(**{v: fwm.var_param_types[v]})

    def _inherit_em_variables(self):
        """
        Retrieves variables from error model and adds them
        to this objects's own set of variables
        """
        em = self._error_model
        for v in em._original_variables:
            if not v == 'mock_data':            
                if v in em.parameters:
                    self._original_variables.update({v})
                else:
                    self._register_variable(v, differentiable=v in
                                            em.differentiable_variables)
                self.update_var_param_types(**{v: em.var_param_types[v]})        

    def _inherit_variables(self):
        """
        Retrieves variables from forward and error model and adds them
        to this objects's own set of variables
        """
        self._inherit_fwm_variables()
        self._inherit_em_variables()
        
    def _setup_parameters(self):
        """
        Sets up parameters matching the parameters in the forward
        and error model and binds the latter to the newly created ones
        """
        for component in (self._forward_model, self._error_model):
            for p in component.get_params():
                self._register(p.name)
                self[p.name] = p.__class__(p.value, p.name, self[p.name])
                p.bind_to(self[p.name])

    def _setup_fixed_variable_parameters(self):
        """
        Sets up fixed variables as parameters, matching the fixed variables
        in the forward and error model and binds the latter to the newly
        created ones
        """
        for model in (self._forward_model, self._error_model):
            for p in model.get_params():
                var_param_type = model.var_param_types[p.name]
                model[p.name] = var_param_type(self[p.name].value, p.name)
                model[p.name].bind_to(self[p.name])

    @property
    def forward_model(self):
        """
        Returns the forward model

        :returns: the forward model of this likelihood object
        :rtype: :class:`.AbstractForwardModel`
        """
        return self._forward_model

    @property
    def error_model(self):
        """
        Returns the error model

        :returns: the error model of this likelihood object
        :rtype: :class:`.AbstractErrorModel`
        """
        return self._error_model

    def _split_variables(self, variables):
        """
        Splits up variables into variables of the forward and
        of the error model

        :param variables: key / value pairs of variables
        :type variables: dict

        :returns: separate kev / value pairs for forward and
                  error model variables
        :rtype: (dict, dict)
        """
        fwm_variables = {v: variables[v] for v in variables
                         if v in self.forward_model.variables}
        em_variables = {v: variables[v] for v in variables
                        if v in self.error_model.variables}

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
        
        return copy

    def conditional_factory(self, **fixed_vars):

        fwm = self.forward_model.clone()
        fwm.fix_variables(**fwm._get_variables_intersection(fixed_vars))
        old_em = self.error_model
        variables_intersection = old_em._get_variables_intersection(fixed_vars)
        em = old_em.conditional_factory(**variables_intersection)
        result = self.__class__(self.name, fwm, em)

        return result            

