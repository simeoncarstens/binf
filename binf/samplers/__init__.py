"""
This module contains samplers used in ISD.
We should try to reuse the samplers in CSB.
"""

from csb.statistics.samplers import State
from csb.statistics.samplers.mc.singlechain import AbstractSingleChainMC


class ISDState(object):

    def __init__(self, variables={}, momenta={}):

        self._variables = {}
        self._momenta = {}
        
        self.update_variables(**variables)
        self.update_momenta(**momenta)

    @property
    def variables(self):
        return self._variables.copy()

    def update_variables(self, **variables):
        self._variables.update(variables)
        
    @property
    def momenta(self):
        return self._momenta.copy()
    
    def update_momenta(self, **momenta):
        self._momenta.update(momenta)
    


class AbstractISD2SingleChainMC(AbstractSingleChainMC):

    def __init__(self, pdf, state, temperature=1.0):

        super(AbstractISD2SingleChainMC, self).__init__(pdf,
                                                        State(state.position),
                                                        temperature)
        
    @AbstractSingleChainMC.state.setter
    def state(self, value):
        if not isinstance(value, State):
            value = State(value)
            
        self._state = value

    def sample(self):

        res = super(AbstractISD2SingleChainMC, self).sample()

        return res.position
