"""
Gibbs sampler
"""

import numpy

from csb.statistics.samplers.mc.singlechain import AbstractSingleChainMC


class GibbsSampler(AbstractSingleChainMC):

    def __init__(self, pdf, state, subsamplers):

        self._state = None
        self._pdf = None
        self._subsamplers = {}
        self._conditional_pdfs = {}

        self._state = state
        self._pdf = pdf
        self._subsamplers = subsamplers

        self._setup_conditional_pdfs()

    def _setup_conditional_pdfs(self):

        for var in self.state.variables.keys():
            fixed_vars = {x: self.state.variables[x] for x in self.state.variables
                          if not x == var}
            self._conditional_pdfs.update({var: self._pdf.conditional_factory(**fixed_vars)})
            self.subsamplers[var].pdf = self._conditional_pdfs[var]

    def _update_conditional_pdf_params(self):

        for pdf in self._conditional_pdfs.values():
            for param in pdf.parameters:
                if param in self._state.variables.keys():
                    pdf[param] = self._state.variables[param]
        
    def _checkstate(self, state):

        if not type(state) == dict:
            raise TypeError(state)

    @property
    def state(self):
        return self._state

    @property
    def subsamplers(self):
        return self._subsamplers

    def update_samplers(self, **samplers):
        self._subsamplers.update(**samplers)

    def sample(self):

        for var in self._pdf.variables:
            self._update_conditional_pdf_params()
            new = self.subsamplers[var].sample()
            self._state.update_variables(**{var: new})

        return self._state

    def _calc_pacc():
        pass

    def _propose():
        pass


if __name__ == '__main__':

    from isd2.pdf import AbstractISDPDF
    from isd2.samplers import ISDState
    from csb.statistics.pdf.parameterized import ParameterValueError, Parameter
    from copy import deepcopy

    import matplotlib.pyplot as plt
    

    class Gauss2D(AbstractISDPDF):

        def __init__(self, k1, k2):

            super(Gauss2D, self).__init__('Gauss2D')

            self._register('k1')
            self._register('k2')

            self.set_params(k1=k1, k2=k2)

            self._register_variable('x')
            self._register_variable('y')

        def log_prob(self, x, y):

            return -0.5 * self['k1'] * x.value ** 2 - 0.5 * self['k2'] * y.value ** 2

        def _validate(self, param, value):

            try:
                float(value.value)
            except TypeError:
                raise ParameterValueError(param, value)

        def clone(self):

            from copy import deepcopy

            return deepcopy(self)

            
    class FakeSampler(object):

        def __init__(self, k, var_name):

            self.k = k
            self.var_name = var_name

        def sample(self):

            return Parameter(numpy.random.normal(scale=1.0 / numpy.sqrt(self.k)), self.var_name)


    k1 = 1.0
    k2 = 0.3

    pdf = Gauss2D(Parameter(k1, 'k1'), Parameter(k2, 'k2'))
    subsamplers = {'x': FakeSampler(k1, 'x'), 'y': FakeSampler(k2, 'y')}

    gipsstate = ISDState({'x': Parameter(5.0, 'x'), 'y': Parameter(2.0, 'y')})
    gips = GibbsSampler(pdf, gipsstate, subsamplers)

    samples = []

    for i in range(5000):
        samples.append(deepcopy(gips.sample()))

    xses = [x.variables['x'].value for x in samples]
    yses = [x.variables['y'].value for x in samples]


    ax = plt.figure()

    ax.add_subplot(221)
    plt.hist([xses, numpy.random.normal(size=5000, scale=1.0 / numpy.sqrt(k1))], 
             bins=40, label=['sampled', 'numpy'])
    plt.legend()
    plt.plot()

    ax.add_subplot(222)
    plt.hist([yses, numpy.random.normal(size=5000, scale=1.0 / numpy.sqrt(k2))], 
             bins=40, label=['sampled', 'numpy'])
    plt.legend()
    plt.plot()

    axis = ax.add_subplot(223)
    h, xedges, yedges = numpy.histogram2d(xses, yses, bins=40, range=((-5, 5), (-5, 5)))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(h, extent=extent, interpolation='nearest')
    plt.show()
