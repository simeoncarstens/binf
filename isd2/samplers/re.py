'''
Replica Exchange
'''

from abc import ABCMeta, abstractmethod

from mpsampling import MPReplicaExchangeMC
from csb.statistics.samplers.mc.multichain import AlternatingAdjacentSwapScheme, RESwapParameterInfo, ReplicaExchangeMC

from isd2.samplers.pdfwrapper import PDFWrapper


class AbstractISD2MPRE(MPReplicaExchangeMC):

    def __init__(self, pdf, sampler_params, schedule, swap_interval=5, n_processes=None):

        self._samples = []

        self._pdf = PDFWrapper(pdf)
        samplers = [self._sampler_factory(p) for p in sampler_params]
        param_infos = [RESwapParameterInfo(samplers[i], samplers[i+1])
                       for i in xrange(len(samplers) - 1)]

        n_procs = len(sampler_params) if n_processes is None else n_processes
        super(AbstractISD2MPRE, self).__init__(samplers, param_infos, n_procs)

        self._swap_scheme = AlternatingAdjacentSwapScheme(self)
        self._swap_interval = swap_interval
        self._schedule = schedule

        self._sample_counter = 0

    def _bind_sampler_params(self):

        for s in self._samplers:
            for p in s._pdf.isd2pdf.parameters:
                s._pdf.isd2pdf[p.name].bind_to(self._pdf[p.name])
                
    def _restore_schedule(self):

        for i, s in enumerate(self._samplers):
            for p in self._schedule:
                s._pdf.isd2pdf[p].set(self._schedule[p][i])

    # def _update_pdfs(self):

    #     for s in self._samplers:
    #         for p in self.pdf.parameters:
    #             s._pdf.isd2pdf[p].set(self.pdf[p].value)

    def sample(self):

        if self._sample_counter % self._swap_interval == 0 and self._sample_counter > 0:
            self._swap_scheme.swap_all()
            res = [self.state]
        else:
            res = super(AbstractISD2MPRE, self).sample()

        self._samples.append(res)

        self._sample_counter += 1

        return res[0][0].position

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

    def _create_sample_request(self, sampler, n):

        from mpsampling import NSampleRequest
        
        req = NSampleRequest(sampler.state, n, sampler.timestep)
        # try:
        #     req.k2 = self._pdf['k2'].value if isinstance(self, AbstractISDPDF) else self._pdf.isd2pdf['k2'].value
        # except KeyError, ParameterNotFoundError:
        #     pass

        req.pdf_parameters = {name: param.value for name, param in sampler._pdf.isd2pdf._params}
        
        return req


class HMCISD2MPRE(AbstractISD2MPRE):

    def _sampler_factory(self, param):

        from isd2.samplers.hmc import ISD2MPFastHMCSampler
        
        return ISD2MPFastHMCSampler(pdf=param.pdf, state=param.state, timestep=param.timestep, 
                                    nsteps=param.nsteps)


class AbstractISD2RE(ReplicaExchangeMC):

    def __init__(self, pdf, sampler_params, swap_interval=5):

        self._samples = []

        self._pdf = PDFWrapper(pdf)
        samplers = [self._sampler_factory(p) for p in sampler_params]
        param_infos = [RESwapParameterInfo(samplers[i], samplers[i+1])
                       for i in xrange(len(samplers) - 1)]

        super(AbstractISD2RE, self).__init__(samplers, param_infos)

        self._swap_scheme = AlternatingAdjacentSwapScheme(self)
        self._swap_interval = swap_interval

        self._sample_counter = 0

    def sample(self):

        self._update_sampler_pdf_params(**{'k2': self._pdf['k2'].value})

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

        from isd2.pdf import AbstractISDPDF
        wrapped_pdf = value if isinstance(value, AbstractISDPDF) else PDFWrapper(value)
        self._set_sampler_pdfs(wrapped_pdf)
        self._pdf = wrapped_pdf

    def _set_sampler_pdfs(self, wrapped_pdf):

        for s in self._samplers:
            s.pdf = wrapped_pdf

    def _update_sampler_pdf_params(self, **params):

        for s in self._samplers:
            s.update_pdf_params(**params)


class HMCISD2RE(AbstractISD2RE):

    def _sampler_factory(self, param):

        from isd2.samplers.hmc import ISD2HMCSampler
        
        return ISD2HMCSampler(pdf=param.pdf, state=param.state, timestep=param.timestep, 
                                    nsteps=param.nsteps)

