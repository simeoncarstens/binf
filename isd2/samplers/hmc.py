'''
HMC sampler
'''

import numpy

from isd2.samplers.pdfwrapper import PDFWrapper

from csb.statistics.samplers import State
from csb.statistics.samplers.mc.singlechain import HMCSampler
from csb.numeric.integrators import FastLeapFrog

from mpsampling import MPFastHMCSampler


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

        from isd2.pdf import AbstractISDPDF
        
        wrapped_pdf = pdf if not isinstance(pdf, AbstractISDPDF) else PDFWrapper(pdf)
        wrapped_state = state if 'position' in dir(state) else State(state)
        super(ISD2HMCSampler, self).__init__(wrapped_pdf, wrapped_state, wrapped_pdf.gradient, 
                                             timestep, nsteps, mass_matrix, integrator, temperature)

    def sample(self):

        res = super(ISD2HMCSampler, self).sample()

        return res

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

        
class ISD2MPFastHMCSampler(MPFastHMCSampler):

    def __init__(self, pdf, state, timestep, nsteps,
                 integrator=FastLeapFrog, temperature=1.0):

        wrapped_pdf = PDFWrapper(pdf)
        super(ISD2MPFastHMCSampler, self).__init__(wrapped_pdf, State(state), wrapped_pdf.gradient, 
                                                   timestep, nsteps, integrator, temperature)

        self.mpinit()

    def sample(self, sample_request):

        from mpsampling import SamplerStats, AbstractMPSingleChainMC, NSampleResult
        
        numpy.random.seed()
        self.state = sample_request.state
        self.timestep = sample_request.timestep

        self.update_pdf_params(**sample_request.pdf_parameters)

        samples = []
        for i in range(sample_request.n):
            sample = super(AbstractMPSingleChainMC, self).sample()
            samples.append((sample, self._last_move_accepted))
            
        sample_result = NSampleResult(samples, SamplerStats(self._nmoves,
                                                            [x[1] for x in samples]))
                
        self.client_pipe_end.send(sample_result)
        
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


class HMCParameterInfo(object):

    def __init__(self, **params):

        self.__dict__.update(**params)
