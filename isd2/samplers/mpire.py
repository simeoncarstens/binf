from abc import ABCMeta
from mpi4py import MPI

from mpsampling_p2p import AbstractMPIReplica, MPIReplicaExchangeMC, GetStateRequest, AbstractReplicaRequest, ExchangeRequest


class UpdatePDFParamsRequest(AbstractReplicaRequest):

    def __init__(self, parameters):
        
        self.parameters = parameters


class ISDMPIReplica(AbstractMPIReplica):

    def __init__(self, state, pdf, pdf_params, 
                 sampler_class, sampler_params, id):

        super(ISDMPIReplica, self).__init__(state, pdf, pdf_params, 
                                            sampler_class, sampler_params, id)

        self._request_processing_table.update(UpdatePDFParamsRequest='self._update_pdf_params({})')

    def _update_pdf_params(self, request):

        for name, value in request.parameters.items():
            self.pdf.isd2pdf[name].set(value)

    def _setup_pdf(self):

        for name, value in self.pdf_params.items():
            self.pdf[name].set(value)


class MPIISD2RE(MPIReplicaExchangeMC):

    __metaclass__ = ABCMeta
    
    def __init__(self, pdf, schedule, swap_interval, target_replica_id, n_replicas, id_offset=1):

        super(MPIISD2RE, self).__init__(id_offset=id_offset, n_replicas=n_replicas)

        self._pdf = pdf
        self._schedule = schedule

        self._sample_counter = 0
        self._target_replica_id = target_replica_id
        self._swap_interval = swap_interval
                
    def sample(self):

        self._update_sampler_pdf_params()
        if self._sample_counter % self._swap_interval == 0 and self._sample_counter > 0:
            swap_list = self._calculate_swap_list(self._sample_counter)
            results = self._trigger_exchanges(swap_list)
            self._update_stats(results)
            self._send_border_replica_sample_requests(swap_list)
        else:
            self._trigger_normal_sampling()

        res = self._get_target_replica_state()
        self.state = res.position

        self._sample_counter += 1

        return res.position

    @property
    def pdf(self):
        return self._pdf
    @pdf.setter
    def pdf(self, value):

        self._pdf = value

    def _update_sampler_pdf_params(self):

        for i in xrange(self._n_replicas):
            request = UpdatePDFParamsRequest(parameters={name: self.pdf[name].value 
                                                         for name in self.pdf.parameters 
                                                         if not name in self._schedule})
            self.comm.send(request, dest=self.id_offset + i)

    def _get_target_replica_state(self):

        request = GetStateRequest(self.rank)
        self.comm.send(request, dest=self.id_offset + self._target_replica_id)
        state = self.comm.recv(source=self.id_offset + self._target_replica_id)

        return state


class MPIGibbsISD2RE(MPIISD2RE):

    __metaclass__ = ABCMeta
    
    def sample(self):

        ## Code duplication... needs to be refactored
        
        if self._sample_counter % self._swap_interval == 0 and self._sample_counter > 0:
            swap_list = self._calculate_swap_list(self._sample_counter)
            results = self._trigger_exchanges(swap_list)
            self._update_stats(results)
            self._send_border_replica_sample_requests(swap_list)
        else:
            self._trigger_normal_sampling()

        res = self._get_target_replica_state()
        self.state = res

        self._sample_counter += 1

        return res

    def _update_sampler_pdf_params(self):
        pass

    def _send_exchange_requests(self, swap_list):
        
        for i in swap_list:
            self.comm.send(ExchangeRequest(self.id_offset+i+1, 'MPIGibbsSimpleReplicaExchanger'), 
                           dest=self.id_offset+i)


class ISDGibbsMPIReplica(ISDMPIReplica):
    
    def _send_energy(self, request):

        state = None
        if request.state is None:
            state = self.state
        else:
            state = request.state
            
        self.comm.send(-self.pdf.isd2pdf.log_prob(**state.variables),
                       dest=request.requesting_replica_id)
