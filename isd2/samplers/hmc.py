'''
HMC sampler
'''

import numpy

from isd2.samplers.pdfwrapper import PDFWrapper

from csb.statistics.samplers import State
from csb.statistics.samplers.mc.singlechain import HMCSampler
from csb.numeric.integrators import FastLeapFrog


class ISD2HMCSampler(HMCSampler):
    '''
    It would be nice to subclass all ISD2 samplers from isd2.samplers.AbstractISD2SingleChainMC,
    but obviously also from their corresponding CSB classes, e.g. this class should
    be subclassed from both AbstractISD2SingleChainMC and 
    csb.statistics.samplers.mc.singlechain.HMCSampler, resulting in diamond inheritance,
    which is bad.
    '''

    def __init__(self, pdf, state, timestep, nsteps,
                 mass_matrix=None, integrator=FastLeapFrog, temperature=1.0):

        wrapped_pdf = PDFWrapper(pdf)
        super(ISD2HMCSampler, self).__init__(wrapped_pdf, State(state), wrapped_pdf.gradient, 
                                             timestep, nsteps, mass_matrix, integrator, temperature)

    def sample(self):

        res = super(ISD2HMCSampler, self).sample()

        return res.position

    @property
    def pdf(self):
        return self._pdf
    @pdf.setter
    def pdf(self, value):
        wrapped_pdf = PDFWrapper(value)
        self._pdf = wrapped_pdf
        self._gradient = wrapped_pdf.gradient
        self._propagator._gradient = wrapped_pdf.gradient
        self._propagator._integrator._gradient = wrapped_pdf.gradient
    

class HMCSampler2(object):
    '''
    Not working, because it's not finished.
    '''
    
    def __init__(self, state, pdf, timestep, nsteps, variable_name):

        raise NotImplementedError

        self.state = state
        self.pdf = pdf
        self.timestep = timestep
        self.nsteps = nsteps
        self.variable_name = variable_name

        self.n_accepted = 0
        self.n_total = 0

    def sample(self):

        from copy import deepcopy

        vn = self.variable_name

        def H(s):
            # d = {vn: s.variables[
            return 0.5 * numpy.sum(s.momenta[vn] ** 2) \
                   - self.pdf.log_prob(vn=s.variables[vn].value)

        s = deepcopy(self.state)
        s._momenta[vn] = numpy.random.normal(len(s.variables[vn].value))

        H_old = H(s)

        x = s._variables[vn]._value
        p = s._momenta[vn]
        
        p -= 0.5 * self.timestep * self.pdf.gradient(vn=x)

        for i in xrange(nsteps - 1):
            x += self.timestep * p
            p -= self.timestep * self.pdf.gradient(vn=x)

        x += self.timestep * p
        p -= 0.5 * self.timestep * self.pdf.gradient(vn=x)

        H_new = H(s)

        if numpy.random.uniform() < numpy.exp(-H_new + H_old):
            self.n_accepted += 1
            self.state.variables[vn].set(s.variables[vn].value)

        return self.state
    
    @property
    def acceptance_rate(self):

        return float(self.n_accepted) / self.n_total

