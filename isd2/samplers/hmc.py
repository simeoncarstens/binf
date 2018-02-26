'''
HMC sampler
'''

from collections import namedtuple

import numpy

from isd2.samplers.pdfwrapper import PDFWrapper

from csb.statistics.samplers import State
from csb.statistics.samplers.mc.singlechain import HMCSampler
from csb.numeric.integrators import FastLeapFrog

from fastcode import FastHMCSampler


HMCSampleStats = namedtuple('HMCSampleStats', 'accepted stepsize')

class NoMMHMCSampler(HMCSampler):

    def __init__(self, pdf, state, gradient, timestep, nsteps,
                 mass_matrix=None, integrator=FastLeapFrog, temperature=1.):
        
        super(HMCSampler, self).__init__(pdf, state, temperature)
        
        self._timestep = None
        self.timestep = timestep
        self._nsteps = None
        self.nsteps = nsteps
        self._mass_matrix = None
        self._momentum_covariance_matrix = None
        self._integrator = integrator
        self._gradient = gradient
        self._propagator = self._propagator_factory()

    def _propagator_factory(self):
        """
        Factory which produces a L{MDPropagator} object initialized with
        the MD parameters given in __init__().

        @return: L{MDPropagator} instance
        @rtype: L{MDPropagator}
        """
        import numpy as np
        from csb.statistics.samplers.mc.propagators import MDPropagator
        return MDPropagator(self._gradient, self._timestep,
                            mass_matrix=np.ones(2),
                            integrator=self._integrator)

class AuxiliarySamplerObject(object):
    '''
    TODO: This so needs to be refactored and made cleaner...
    '''

    def _adapt_timestep(self):

        if self.last_move_accepted:
            self.timestep *= self.adaption_uprate
        else:
            self.timestep *= self.adaption_downrate

    @property
    def pdf(self):
        return self._pdf
    @pdf.setter
    def pdf(self, value):
        wrapped_pdf = PDFWrapper(value)
        self._pdf = wrapped_pdf
        self._update_gradients()
        
    def _update_gradients(self):

        self._gradient = self._pdf.gradient
        self._propagator._gradient = self._pdf.gradient
        try:
            self._propagator._integrator._gradient = self._pdf.gradient
        except AttributeError:
            pass

    def update_pdf_params(self, **params):

        for p in params:
            self._pdf.isd2pdf[p].set(params[p])

    def get_last_draw_stats(self):

        return {self.variable_name(): HMCSampleStats(self.last_move_accepted,
                                                     self.timestep)}

    @property
    def sampling_stats(self):

        from collections import OrderedDict

        return OrderedDict(**{'HMC acceptance rate': self.acceptance_rate, 
                              'HMC timestep': self.timestep, 
                              'HMC pseudo-energy': self.energy})    


class ISD2HMCSampler(NoMMHMCSampler, AuxiliarySamplerObject):
    '''
    It would be nice to subclass all ISD2 samplers from isd2.samplers.AbstractISD2SingleChainMC,
    but obviously also ritance,
    which is bad.
    '''

    def __init__(self, pdf, state, timestep, nsteps, timestep_adaption_limit=0,
                 adaption_uprate=1.05, adaption_downrate=0.95, variable_name=None):

        from isd2.pdf import AbstractISDPDF
        
        wrapped_pdf = pdf if not isinstance(pdf, AbstractISDPDF) else PDFWrapper(pdf)
        wrapped_state = state if 'position' in dir(state) else State(state)
        super(ISD2HMCSampler, self).__init__(wrapped_pdf, wrapped_state, wrapped_pdf.gradient, 
                                                 timestep, nsteps)

        self.timestep_adaption_limit = timestep_adaption_limit
        self.adaption_uprate = adaption_uprate
        self.adaption_downrate = adaption_downrate

        self._variable_name = variable_name
        self.counter = 0

    def variable_name(self):
        return 'HMC' if self._variable_name is None else self._variable_name

    def sample(self):
        
        res = super(ISD2HMCSampler, self).sample()

        if self.counter < self.timestep_adaption_limit:
            self._adapt_timestep()
        self.counter += 1
        
        return res.position

    @property
    def sampling_stats(self):

        from collections import OrderedDict

        return OrderedDict(**{'{} acceptance rate'.format(self.variable_name):
                                self.acceptance_rate, 
                              '{} timestep'.format(self.variable_name):
                                self.timestep, 
                              '{} pseudo-energy'.format(self.variable_name):
                                self.energy})

class HMCParameterInfo(object):

    def __init__(self, **params):

        self.__dict__.update(**params)


class FastHMCSampler(ISD2HMCSampler, AuxiliarySamplerObject):

    def sample(self):

        import numpy as np
        from csb.numeric import exp

        q = self.state.copy() if not 'position' in dir(self.state) else self.state._position.copy()
        p = np.random.normal(size=q.shape)
        E_before = -self.pdf.log_prob(q) + 0.5 * np.sum(p ** 2)

        p -= 0.5 * self.timestep * self._gradient(q)

        for i in range(self.nsteps-1):
            q += p * self.timestep
            p -= self.timestep * self._gradient(q)

        q += p * self.timestep
        p -= 0.5 * self.timestep * self._gradient(q)

        E_after = -self.pdf.log_prob(q) + 0.5 * np.sum(p ** 2)

        acc = np.random.uniform() < exp(-(E_after - E_before))

        self._update_statistics(acc)
        self._last_move_accepted = acc

        if self.counter < self.timestep_adaption_limit:
            self._adapt_timestep()
        self.counter += 1

        if acc:
            self._state = q
            return q.copy()
        else:
            return self._state.copy() if not 'position' in dir(self._state) else self._state.position.copy()
        
