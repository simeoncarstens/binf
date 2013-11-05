"""
This module contains samplers used in ISD.
We should try to reuse the samplers in CSB.
"""

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
    
