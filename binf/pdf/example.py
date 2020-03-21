"""
Example illustrating the use of the Parameter class
"""

from csb.statistics.pdf import AbstractDensity, ParameterNotFoundError
from csb.statistics.pdf.parameterized import Parameter, ParameterValueError
from csb.core import typedproperty, iterable

from binf.pdf.parameters import FancyGaussian

import numpy as np
import numpy

class Likelihood(AbstractDensity):
    """
    A probability density function with data and parameters.
    """
    
    @typedproperty(np.ndarray)
    def data():
        pass

class NormalLikelihood(FancyGaussian):

    @typedproperty(numpy.ndarray)
    def data():
        pass

    @property
    def mock(self):

        mock = L._params.get('mock',0.)

        if isinstance(mock,Parameter):
            mock = mock.value

        return mock
        
    def log_prob(self):
        return super(NormalLikelihood, self).log_prob(self.data-self.mock).sum()

class LinearModel(Parameter):

    def __init__(self, basis, input, value=None):

        value = value if value is not None else np.zeros(len(input))

        self._basis = basis
        self._input = np.array(input)

        super(LinearModel, self).__init__(value=value)

        self._design_matrix = None
        self._update_design_matrix()

    @property
    def x(self):
        return self._input

    @property
    def y(self):
        return self.value
        
    def _update_design_matrix(self):

        self._design_matrix = np.array([f(self.x) for f in self._basis]).T
        
    def _validate(self, value):

        if iterable(value) and len(value) == len(self._input):
            return np.array(value)
        else:
            raise ValueError()

    def _compute(self, base_value):

        if not len(base_value) == len(self._basis):
            raise ValueError()
        
        return np.inner(self._design_matrix, base_value)

class Coefficients(Parameter):

    def __init__(self, n):

        super(Coefficients, self).__init__(np.zeros(n))

    def __len__(self):
        return len(self._value)

    def _validate(self, value):

        try:
            return numpy.array(value)
        except(ValueError, TypeError):
            raise ParameterValueError(self.name, value)

if __name__ == '__main__':

    n = 100

    ## straight-line fitting

    basis = (lambda x: 0 * x + 1, lambda x: x)
    x = np.linspace(-10., 10., n)
    noise = np.random.standard_normal(n)
    model = LinearModel(basis, x)
    theta = Coefficients(len(basis))
    model.set(noise)
    model.bind_to(theta)
    theta.set([0.,1.])
    y = model.value
    
    L = NormalLikelihood()
    L.data = y + noise
    L._register('mock')
    L.set_params(mock=model)

    print(L.log_prob())

    ## Monte Carlo

    stepsize = 1e-1
    p = np.array([5.,5.]) #np.random.random(len(theta))
    theta.set(p)
    P = L.log_prob()
    a = 0
    M = 10000
    
    samples = np.zeros((M,len(p)+1))

    def energy(x):
        theta.set(x)
        return -L.log_prob()

    for i in range(M):

        q = p + stepsize * (np.random.random(len(p))-0.5)
        theta.set(q)
        Q = L.log_prob()

        if np.log(np.random.random()) < Q - P:
            p, P = q, Q
            a += 1

        theta.set(p)
        chi2 = np.linalg.norm(L.data - L.mock)**2
        L.precision = np.random.gamma(0.5 * n) / (0.5 * chi2)
        P = L.log_prob()
        
        samples[i,:2] = p
        samples[i,-1] = L.precision
