"""
Gibbs sampler implementations
"""

import numpy

from csb.statistics.samplers import State
from csb.statistics.samplers.mc.singlechain import AbstractSingleChainMC


class GibbsSampler(AbstractSingleChainMC):

    def __init__(self, pdf, state, subsamplers):
        """
        Implements a Gibbs sampler

        :param pdf: PDF object representing the pdf this sampler
                    is supposed to sample from
        :type pdf: :class:`.AbstractBinfPDF`

        :param state: initial state
        :type state: :class:`.BinfState`

        :param subsamplers: subsamplers for each variable of the
                            PDF object
        :type subsamplers: dict
        """
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
        """
        Sets up the conditional PDFs from the full PDF 
        """
        for var in list(self.state.variables.keys()):
            fixed_vars = {x: self.state.variables[x]
                          for x in self.state.variables if not x == var}
            fixed_vars.update(**{x: self._pdf[x].value
                                 for x in self._pdf.parameters 
                                 if x in self._pdf._original_variables})
            cond_pdf = self._pdf.conditional_factory(**fixed_vars)
            self._conditional_pdfs.update({var: cond_pdf})
            self.subsamplers[var].pdf = self._conditional_pdfs[var]

    def _update_conditional_pdf_params(self):
        """
        Updates parameters of the conditional PDFs to values set
        in this object's PDF
        """
        for pdf in list(self._conditional_pdfs.values()):
            for param in pdf.parameters:
                if param in list(self._state.variables.keys()):
                    pdf[param].set(self._state.variables[param])
        
    def _checkstate(self, state):
        """
        Checks whether the state is a dict
        """
        if not type(state) == dict:
            raise TypeError(state)

    @property
    def pdf(self):
        """
        Returns the PDF object this sampler samples from

        :returns: PDF object
        :rtype: :class:`.AbstractBinfPDF`
        """
        return self._pdf
    @pdf.setter
    def pdf(self, value):
        """
        Sets the PDF this sampler samples from and refreshes
        the conditional PDFs
        """
        self._pdf = value
        self._setup_conditional_pdfs()

    @property
    def state(self):
        """
        Returns the current state of this sampler

        :returns: current state
        :rtype: :class:`.BinfState`
        """
        return self._state

    @property
    def subsamplers(self):
        """
        Returns the subsamplers the Gibbs sampler iterates over

        :returns: subsamplers for each variable
        :rtype: dict
        """
        return self._subsamplers

    def update_samplers(self, **samplers):
        """
        Updates the subsamplers
        """
        self._subsamplers.update(**samplers)

    def _update_subsampler_states(self):
        """
        Updates subsampler states, e.g., after a replica exchange swap
        """
        from csb.statistics.samplers.mc import AbstractMC

        for variable in self.state.variables:
            if isinstance(self.subsamplers[variable], AbstractMC):
                self.subsamplers[variable].state = State(self.state.variables[variable])
            else:
                self.subsamplers[variable].state = self.state.variables[variable]

    def _update_state(self, **variables):
        """
        Updates this object's state's variables to new values
        """
        for variable, new_value in list(variables.items()):
            if type(new_value) == State:
                new_value = new_value.position
            self._state.update_variables(**{variable: new_value})
                
    def sample(self):
        """
        Performs one iteration of the Gibbs sampling scheme

        :returns: a state
        :rtype: :class:`.BinfState`
        """
        ## needed for RE
        self._update_subsampler_states()
        
        for var in sorted(list(self._pdf.variables)):
            self._update_conditional_pdf_params()
            new = self.subsamplers[var].sample()
            self._update_state(**{var: new})

        return self._state

    def _calc_pacc():
        """
        Not applicable
        """
        pass

    def _propose():
        """
        Not applicable
        """
        pass

    @property
    def last_draw_stats(self):
        """
        Returns information about most recent move for each subsampler

        :returns: information about most recent move for each subsampler
        :rtype: dict
        """
        return {k: v.last_draw_stats[k] for k, v in list(self.subsamplers.items()) 
                if getattr(v, 'last_draw_stats', None) is not None}

    @property
    def sampling_stats(self):
        """
        Sampling statistics, consisting of sampler statistics of the
        subsamplers

        :returns: all subsampler sampling statistics
        :rtype: OrderedDict
        """
        from collections import OrderedDict

        ss = [s for s in list(self.subsamplers.values()) if 'sampling_stats' in dir(s)]

        return OrderedDict(**{key: value for sampler in ss 
                              for key, value in list(sampler.sampling_stats.items())})
    
