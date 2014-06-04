'''
Replica Exchange
'''

from abc import ABCMeta, abstractmethod

from csb.statistics.samplers.mc.multichain import AlternatingAdjacentSwapScheme, RESwapParameterInfo
from mpsampling_new import SimpleReplicaExchangeMC as ReplicaExchangeMC, MPReplicaExchangeMC, MPSampleCommunicator, AbstractReplicaExchangeMC

from isd2.samplers.pdfwrapper import PDFWrapper


class AbstractISD2RE(AbstractReplicaExchangeMC):

    def __init__(self, pdf, sampler_params, schedule, swap_interval=5, **extra_params):

        self._samples = []

        self._pdf = PDFWrapper(pdf)

        self._setup_csb_re(sampler_params, **extra_params)
        
        self._swap_scheme = AlternatingAdjacentSwapScheme(self)
        self._swap_interval = swap_interval
        self._schedule = schedule

        self._sample_counter = 0

    def _setup_csb_re(self, sampler_params, **extra_params):

        samplers = [self._sampler_factory(p) for p in sampler_params]
        param_infos = [RESwapParameterInfo(samplers[i], samplers[i+1])
                       for i in xrange(len(samplers) - 1)]

        super(AbstractISD2RE, self).__init__(samplers, param_infos)

    def _bind_sampler_params(self):

        for s in self._samplers:
            for p in s._pdf.isd2pdf.parameters:
                if p in self._schedule:
                    continue
                s._pdf.isd2pdf[p].bind_to(self._pdf[p])
                
    def _restore_schedule(self):

        for i, s in enumerate(self._samplers):
            for p in self._schedule:
                s._pdf.isd2pdf[p].set(self._schedule[p][i])

    def sample(self):

        if self._sample_counter % self._swap_interval == 0 and self._sample_counter > 0:
            self._swap_scheme.swap_all()
            res = self.state
        else:
            res = super(AbstractISD2RE, self).sample()

        self._samples.append(res)

        self._sample_counter += 1

        return res[0].position

    @abstractmethod
    def _sampler_factory(self, param):
        pass

    @property
    def pdf(self):
        return self._pdf
    @pdf.setter
    def pdf(self, value):

        self._set_sampler_pdfs(value)
        self._pdf = value
        self._bind_sampler_params()

        self._restore_schedule()

    def _set_sampler_pdfs(self, xpdf):
        
        for s in self._samplers:
            pdf = xpdf.clone()
            s.pdf = pdf

    def _update_sampler_pdf_params(self, **params):

        for s in self._samplers:
            s.update_pdf_params(**params)


class AbstractISD2MPRE(AbstractISD2RE):

    def __init__(self, pdf, sampler_params, schedule, swap_interval=5, n_processes=None):

        self._n_processes = n_processes

        super(AbstractISD2MPRE, self).__init__(pdf, sampler_params, schedule, swap_interval)


class HMCISD2MPSampleCommunicator(MPSampleCommunicator):

    def _create_sample_request(self, sampler):
        
        from mpsampling import NSampleRequest
        
        req = NSampleRequest(sampler.state, 1, sampler.timestep)
        req.pdf_parameters = {name: param.value 
                              for name, param in sampler._pdf.isd2pdf._params.iteritems()}
        
        return req


class HMCISD2MPRE(AbstractISD2MPRE):

    def _sampler_factory(self, param):

        from isd2.samplers.hmc import ISD2MPFastHMCSampler
        
        return ISD2MPFastHMCSampler(pdf=param.pdf, state=param.state, timestep=param.timestep, 
                                    nsteps=param.nsteps)

    def _adapt_sampler_timesteps(self):

        for s in self._samplers:
            if s.last_move_accepted:
                s.timestep *= 1.05
            else:
                s.timestep *= 0.95

    def sample(self):
        
        res = super(HMCISD2MPRE, self).sample()
        
        if True:
            self._adapt_sampler_timesteps()
            
        return res

    def _set_sample_communicator(self):

        self._sample_communicator = HMCISD2MPSampleCommunicator(self, self._n_processes)


class HMCISD2RE(AbstractISD2RE):

    def _sampler_factory(self, param):

        from isd2.samplers.hmc import ISD2HMCSampler
        
        return ISD2HMCSampler(pdf=param.pdf, state=param.state, timestep=param.timestep, 
                              nsteps=param.nsteps)
