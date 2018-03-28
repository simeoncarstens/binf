"""
Gibbs sampler
"""

import numpy

from csb.statistics.samplers import State
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
        self._update_subsampler_states()

    def _setup_conditional_pdfs(self):

        for var in self.state.variables.keys():
            fixed_vars = {x: self.state.variables[x] for x in self.state.variables
                          if not x == var}
            fixed_vars.update(**{x: self._pdf[x].value for x in self._pdf.parameters 
                                 if x in self._pdf._original_variables})
            self._conditional_pdfs.update({var: self._pdf.conditional_factory(**fixed_vars)})
            self.subsamplers[var].pdf = self._conditional_pdfs[var]

    def _update_conditional_pdf_params(self):

        for pdf in self._conditional_pdfs.values():
            for param in pdf.parameters:
                if param in self._state.variables.keys():
                    pdf[param].set(self._state.variables[param])
        
    def _checkstate(self, state):

        if not type(state) == dict:
            raise TypeError(state)

    @property
    def pdf(self):
        return self._pdf
    @pdf.setter
    def pdf(self, value):
        self._pdf = value
        self._setup_conditional_pdfs()

    @property
    def state(self):
        return self._state

    @property
    def subsamplers(self):
        return self._subsamplers

    def update_samplers(self, **samplers):
        self._subsamplers.update(**samplers)

    def _update_subsampler_states(self):

        from csb.statistics.samplers.mc import AbstractMC

        for variable in self.state.variables:
            if isinstance(self.subsamplers[variable], AbstractMC):
                self.subsamplers[variable].state = State(self.state.variables[variable])
            else:
                self.subsamplers[variable].state = self.state.variables[variable]

    def _update_state(self, **variables):

        for variable, new_value in variables.items():
            if type(new_value) == State:
                new_value = new_value.position
            self._state.update_variables(**{variable: new_value})
                
    def sample(self):
        
        ## needed for RE
        self._update_subsampler_states()
        
        for var in sorted(list(self._pdf.variables)):
            self._update_conditional_pdf_params()
            new = self.subsamplers[var].sample()
            self._update_state(**{var: new})

        return self._state

    def _calc_pacc():
        pass

    def _propose():
        pass

    @property
    def last_draw_stats(self):

        return {k: v.last_draw_stats[k] for k, v in self.subsamplers.items() 
                if getattr(v, 'last_draw_stats', None) is not None}

    @property
    def sampling_stats(self):

        from collections import OrderedDict

        ss = [s for s in self.subsamplers.values() if 'sampling_stats' in dir(s)]

        return OrderedDict(**{key: value for sampler in ss 
                              for key, value in sampler.sampling_stats.items()})
    
