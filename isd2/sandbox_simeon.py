"""
Abusing new stuff
to test these interfaces by sampling coefficients of a polynom.
"""

import numpy

from isd2.model.forwardmodels import AbstractDifferentiableForwardModel
from isd2.model.errormodels import GaussianErrorModel
from isd2.pdf import ParameterNotFoundError
from isd2.pdf.posteriors import Posterior, DifferentiablePosterior
from isd2.pdf.priors import AbstractPrior, AbstractDifferentiablePrior
from isd2.pdf.likelihoods import AbstractDifferentiableLikelihood

from csb.numeric import log
from csb.statistics.samplers import State
from csb.statistics.samplers.mc.propagators import RWMCPropagator
from csb.statistics.pdf.parameterized import AbstractParameter, Parameter, ParameterValueError

import matplotlib.pyplot as plt


class ForwardModel(AbstractDifferentiableForwardModel):

    def __init__(self, name, data):

        super(ForwardModel, self).__init__(name, data)

        self._register_variable('coeffs')

    def _validate(self, param, value):

        pass

    def __call__(self, **variables):

        coeffs = variables['coeffs'].value

        t1 = self.data[0] ** 2 * coeffs[0]
        t2 = self.data[0] * coeffs[1] + coeffs[2]
        ys = numpy.sum(numpy.vstack((t1, t2)), 0)

        return numpy.vstack((self.data[0], ys, self.data[1]))

    def jacobi_matrix(self, **variables):

        return numpy.vstack((self.data[0] ** 2, self.data[0], numpy.ones(len(data[0]))))


class ThetaPrior(AbstractPrior):

    def __init__(self, name='theta prior', bounds=None):

        super(ThetaPrior, self).__init__(name)
        
        for i, x in enumerate(bounds):
            self._register(x.name)
            
        self.set_params(*bounds)

        self._register_variable('coeffs')
        
    def log_prob(self, **variables):

        coeffs = variables['coeffs'].value
        
        theta_evals = [float(x > self.get_params()[i].value[0] and x < self.get_params()[i].value[1])
                       for i, x in enumerate(coeffs)]
        
        return log(numpy.multiply.accumulate(theta_evals)[-1])

    def _validate(self, param, value):

        if type(value.value) != tuple:
            raise ParameterTypeError(param, value)
        elif len(value.value) != 2:
            raise ParameterValueError(param, value)
        elif value.value[0] > value.value[1]:
            raise ParameterValueError(param, value, 
                                      "Lower bond must not be higher than the upper bound")


class SigmaPrior(AbstractPrior):

    def __init__(self, name='sigma prior'):

        super(SigmaPrior, self).__init__(name)

        self._register_variable('sigma')

    def log_prob(self, **variables):

        sigma = variables['sigma']

        return log(float(sigma.value > 0.0))
            
            
class MyLikelihood(AbstractDifferentiableLikelihood):

    def __init__(self, forward_model, error_model, name='blabla'):

        super(MyLikelihood, self).__init__(name, forward_model, error_model)

        self._register_variable('coeffs')


class GaussianPrior(AbstractDifferentiablePrior):

    def __init__(self, mu, sigma, name='gaussian prior for coefficients'):

        super(GaussianPrior, self).__init__(name)

        self._register('mu')
        self._register('sigma')
        self.set_params(mu=mu, sigma=sigma)

        self._register_variable('coeffs')

    def log_prob(self, coeffs):

        return -0.5 * sum((coeffs.value - self['mu'].value) ** 2) / (self['sigma'].value ** 2) #- 0.5 * log(2.0 * numpy.pi)

    def gradient(self, coeffs):

        return (coeffs.value - self['mu'].value) / (self['sigma'].value ** 2)

    def _validate(self, param, value):

        # if param != 'sigma' and param != 'mu':
        #     raise ParameterNotFoundError('GaussianPrior allows only for ' + 
        #                                  'one parameter called \'sigma\'')

        if param == 'mu':
            try:
                numpy.array(value.value)
            except TypeError:
                raise ParameterValueError(param, value)

        if param == 'sigma':
            try:
                value = float(value.value)
            except TypeError:
                raise ParameterValueError(param, value)
            if value <= 0.0:
                raise ParameterValueError(param, value, 'sigma has to be >= 0')
            

        
class SamplePDF(object):

    def __init__(self, posterior):

        self.posterior = posterior
        
    def log_prob(self, x):

        return self.posterior.log_prob(coeffs=ArrayParameter(numpy.array(x[:3]), 'coeffs'), sigma=Parameter(x[3], 'sigma'))
    
    
class CoeffsPDF(object):

    def __init__(self, posterior):

        self.posterior = posterior
        
    def log_prob(self, x):

        return self.posterior.log_prob(coeffs=ArrayParameter(x[:3], 'coeffs'))

    def gradient(self, x, t=0.0):

        return self.posterior.gradient(coeffs=ArrayParameter(x[:3], 'coeffs'))


def numerical_gradient(x, E, eps=1e-6):

    res = numpy.zeros(len(x))
    Eold = E(x)
    for i in range(len(x)):
        x[i] += eps
        res[i] += (E(x) - Eold) / eps
        x[i] -= eps

    return res


class ArrayParameter(AbstractParameter):

    def _validate(self, value):
        try:
            return numpy.array(value)
        except(TypeError, ValueError):
            raise ParameterValueError(self.name, value)


class TupleParameter(AbstractParameter):

    def _validate(self, value):
        try:
            return tuple(value)
        except (TypeError, ValueError):
            print self.name, value
            raise ParameterValueError(self.name, value)
        


coeffs = numpy.array([2.0, -1.0, 1.0])
bounds = [TupleParameter((1.0, 3.0), 'bounds_a'), 
          TupleParameter((-2.0, 0.0), 'bounds_b'), 
          TupleParameter((0.0, 2.0), 'bounds_c')]
sigma = 2.0

data = numpy.array([(x, numpy.random.normal(loc=coeffs[0] * x ** 2 + coeffs[1] * x + coeffs[2], 
                                            scale=sigma)) 
                    for x in numpy.linspace(-5, 5, 35)]).swapaxes(0,1)
        
start = {'coeffs': numpy.array([0.0, 0.0, 0.0]), 'sigma': 1.0}

FWM = ForwardModel('polynom', data)
EM = GaussianErrorModel('gaussian', Parameter(start['sigma'], name='sigma'))
L = MyLikelihood(FWM, EM, 'Likiliki')

if False:

    P = ThetaPrior(bounds=bounds)
    SP = SigmaPrior()

    posti = Posterior({L.name: L}, {SP.name: SP, P.name: P})
    
    spdf = SamplePDF(posti)

    
    gen = RWMCPropagator(spdf, stepsize=.02)

    samples = gen.generate(State(numpy.array([2.0, -1.0, 1.0, 1.0])), length=15000, return_trajectory=True)
    
def plot_hists(samples, coeffs):

    histrange = (-2, 2.5)
    bins = 50
    
    ax = plt.figure()
    
    ax.add_subplot(221)
    plt.hist([x.position[0] for x in samples], range=histrange, bins=bins)
    plt.title('a='+str(coeffs[0]))
    
    ax.add_subplot(222)
    plt.hist([x.position[1] for x in samples], range=histrange, bins=bins)
    plt.title('b='+str(coeffs[1]))
    
    ax.add_subplot(223)
    plt.hist([x.position[2] for x in samples], range=histrange, bins=bins)
    plt.title('c='+str(coeffs[2]))

    if len(samples[0].position) == 4:
        ax.add_subplot(224)
        plt.hist([x.position[3] for x in samples], range=histrange, bins=bins)
        plt.title('sigma='+str(sigma))
    
    plt.show()


if True:
    
    mu = ArrayParameter(coeffs[:3], 'mu')
    sigma = Parameter(1.0, 'sigma')
    GP = GaussianPrior(mu, sigma)
    SP = SigmaPrior()
    fullposti = DifferentiablePosterior({L.name: L}, 
                                        {SP.name: SP, GP.name: GP})
    posti = fullposti.conditional_factory(sigma=Parameter(2.0, 'sigma'))

    y = numpy.array([2.1, -2.0, 3.0, 2.0])

    # print L.gradient(coeffs=y[:3])
    # print numerical_gradient(y[:3], lambda z: -L.log_prob(coeffs=z), eps=1e-6)

    # print y

    pdf = CoeffsPDF(posti)

    from csb.statistics.samplers.mc.propagators import HMCPropagator
    
    gen = HMCPropagator(pdf=pdf, gradient=pdf.gradient, timestep=0.02, nsteps=20)

    samples = gen.generate(State(numpy.array([2.0, -1.0, 1.0])), length=2000, return_trajectory=True)
