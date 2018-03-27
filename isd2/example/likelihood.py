import numpy as np

from csb.statistics.pdf.parameterized import Parameter as ScalarParameter

from isd2 import ArrayParameter
from isd2.model.forwardmodels import AbstractForwardModel
from isd2.model.errormodels import AbstractErrorModel
from isd2.pdf import ParameterNotFoundError


class ForwardModel(AbstractForwardModel):

    def __init__(self, xses, polynomial):

        super(ForwardModel, self).__init__('polynomial')
        
        self.xses = xses
        self.polynomial = polynomial
        
        self._register_variable('coefficients', differentiable=True)
        self.update_var_param_types(coefficients=ArrayParameter)
        self._set_original_variables()

    def _evaluate(self, coefficients):

        return self.polynomial(self.xses, coefficients)

    def _evaluate_jacobi_matrix(self, coefficients):

        return np.vstack([self.xses ** i for i in range(len(coefficients))])

    def clone(self):

        copy = self.__class__(self.xses, self.polynomial)
        
        for p in self.parameters:
            if not p in copy.parameters:
                copy._register(p)
                copy[p] = self[p].__class__(self[p].value, p)
                if p in copy.variables:
                    copy._delete_variable(p)                

        return copy


class GaussianErrorModel(AbstractErrorModel):

    def __init__(self, ys):

        super(GaussianErrorModel, self).__init__('error_model')

        self.ys = ys

        self._register_variable('mock_data')
        self._register_variable('precision')
        self.update_var_param_types(mock_data=ArrayParameter, precision=ScalarParameter)
        self._set_original_variables()

    def _evaluate_log_prob(self, mock_data, precision):

        return -0.5 * np.sum((mock_data - self.ys) ** 2) * precision + len(self.ys) * 0.5 * np.log(precision)

    def _evaluate_gradient(self, mock_data, precision):

        return (mock_data - self.ys) * precision

    def clone(self):

        copy = self.__class__(self.ys)
        copy.set_fixed_variables_from_pdf(self)

        return copy

def make_likelihood(xses, ys, polynomial):
    
    from isd2.pdf.likelihoods import Likelihood
    from isd2.example.likelihood import ForwardModel, GaussianErrorModel

    FWM = ForwardModel(xses, polynomial)
    EM = GaussianErrorModel(ys)
    L = Likelihood('points', FWM, EM)

    return L
