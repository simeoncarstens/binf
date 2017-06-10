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


HMCSampleStats = namedtuple('HMCSampleStats', 'accepted total')


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

        return HMCSampleStats(self._accepted, self._nmoves)

    @property
    def sampling_stats(self):

        from collections import OrderedDict

        return OrderedDict(**{'HMC acceptance rate': self.acceptance_rate, 
                              'HMC timestep': self.timestep, 
                              'HMC pseudo-energy': self.energy})
    


class ISD2HMCSampler(HMCSampler, AuxiliarySamplerObject):
    '''
    It would be nice to subclass all ISD2 samplers from isd2.samplers.AbstractISD2SingleChainMC,
    but obviously also from their corresponding CSB classes, e.g. this class should
    be subclassed from both AbstractISD2SingleChainMC and 
    csb.statistics.samplers.mc.singlechain.HMCSampler, resulting in diamond inheritance,
    which is bad.
    '''

    def __init__(self, pdf, state, timestep, nsteps,
                 mass_matrix=None, integrator=FastLeapFrog, temperature=1.0):

        from isd2.pdf import AbstractISDPDF
        
        wrapped_pdf = pdf if not isinstance(pdf, AbstractISDPDF) else PDFWrapper(pdf)
        wrapped_state = state if 'position' in dir(state) else State(state)
        super(ISD2HMCSampler, self).__init__(wrapped_pdf, wrapped_state, wrapped_pdf.gradient, 
                                             timestep, nsteps, mass_matrix, integrator, temperature)

    def sample(self):

        res = super(ISD2HMCSampler, self).sample()
        
        return res


class ISD2FastHMCSampler(FastHMCSampler, AuxiliarySamplerObject):
    '''
    It would be nice to subclass all ISD2 samplers from isd2.samplers.AbstractISD2SingleChainMC,
    but obviously also from their corresponding CSB classes, e.g. this class should
    be subclassed from both AbstractISD2SingleChainMC and 
    csb.statistics.samplers.mc.singlechain.HMCSampler, resulting in diamond inheritance,
    which is bad.
    '''

    def __init__(self, pdf, state, timestep, nsteps, 
                 timestep_adaption=True, adaption_uprate=1.05, adaption_downrate=0.95,
                 variable_name=None):

        from isd2.pdf import AbstractISDPDF
        
        wrapped_pdf = pdf if not isinstance(pdf, AbstractISDPDF) else PDFWrapper(pdf)
        wrapped_state = state if 'position' in dir(state) else State(state)
        super(ISD2FastHMCSampler, self).__init__(wrapped_pdf, wrapped_state, wrapped_pdf.gradient, 
                                                 timestep, nsteps)

        self.timestep_adaption = timestep_adaption
        self.adaption_uprate = adaption_uprate
        self.adaption_downrate = adaption_downrate

        self._variable_name = None

    def variable_name(self):
        return 'HMC' if self._variable_name is None else self._variable_name

    def sample(self):
        
        res = super(ISD2FastHMCSampler, self).sample()

        if self.timestep_adaption:
            self._adapt_timestep()
            
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
