import numpy as np

from csb.statistics.pdf.parameterized import Parameter as ScalarParameter

from binf import ArrayParameter
from binf.model.forwardmodels import AbstractForwardModel
from binf.model.errormodels import AbstractErrorModel
from binf.pdf import ParameterNotFoundError


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
        self._set_parameters(copy)

        return copy


class GaussianErrorModel(AbstractErrorModel):

    def __init__(self, ys):

        super(GaussianErrorModel, self).__init__('error_model')

        self.ys = ys

        self._register_variable('mock_data')
        self._register_variable('precision')
        self.update_var_param_types(mock_data=ArrayParameter,
                                    precision=ScalarParameter)
        self._set_original_variables()

    def _evaluate_log_prob(self, mock_data, precision):

        logZ = len(self.ys) * 0.5 * np.log(precision)
        return -0.5 * np.sum((mock_data - self.ys) ** 2) * precision + logZ

    def _evaluate_gradient(self, mock_data, precision):

        return (mock_data - self.ys) * precision

    def clone(self):

        copy = self.__class__(self.ys)
        copy.set_fixed_variables_from_pdf(self)

        return copy

def make_likelihood(xses, ys, polynomial):
    
    from binf.pdf.likelihoods import Likelihood
    from binf.example.likelihood import ForwardModel, GaussianErrorModel

    FWM = ForwardModel(xses, polynomial)
    EM = GaussianErrorModel(ys)
    L = Likelihood('points', FWM, EM)

    return L
