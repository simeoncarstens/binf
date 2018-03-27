import numpy as np

from csb.statistics.pdf.parameterized import Parameter as ScalarParameter

from isd2 import ArrayParameter

from isd2.pdf.posteriors import Posterior
from isd2.pdf.priors import AbstractPrior

class GammaPrior(AbstractPrior):

    def __init__(self, shape, rate):

        super(GammaPrior, self).__init__('precision_prior')

        self.shape = shape
        self.rate = rate
        
        self._register_variable('precision')
        self.update_var_param_types(precision=ScalarParameter)
        self._set_original_variables()

    def _evaluate_log_prob(self, precision):

        return (self.shape - 1.0) * np.log(precision) - precision * self.rate

    def clone(self):

        copy = self.__class__(self.shape, self.shape)
        copy.set_fixed_variables_from_pdf(self)

        return copy
            

class GaussianPrior(AbstractPrior):

    def __init__(self, means, variances):

        super(GaussianPrior, self).__init__('coefficients_prior')
        
        self._register('means')
        self._register('variances')
        self['means'] = ArrayParameter(means, 'means')
        self['variances'] = ArrayParameter(variances, 'variances')
        self._register_variable('coefficients')
        self.update_var_param_types(coefficients=ArrayParameter)
        self._set_original_variables()

    def _evaluate_log_prob(self, coefficients):

        means = self['means'].value
        variances = self['variances'].value

        return -0.5 * np.sum((coefficients - means) ** 2 / variances)
    
    def _evaluate_gradient(self, **variables):

        coeff = variables[self.variable_name]
        
        return (coeff - self['mu'].value) / self['sigma'].value ** 2

    def clone(self):

        return self.__class__(self['means'].value, self['variances'].value)


def make_priors():
    
    PP = GammaPrior(1.0, 0.2)
    CP = GaussianPrior(means=np.array([2.0, -1.0, 1.0, 0.5]), variances=np.ones(4) * 5)

    return {PP.name: PP, CP.name: CP}
