"""
This module contains implementations of various MCMC samplers
"""

from csb.statistics.samplers import State
from csb.statistics.samplers.mc.singlechain import AbstractSingleChainMC


class ISDState(object):

    def __init__(self, variables={}, momenta={}):
        """
        Supposed to represent a state in a Markov chain, containing
        named and typed variables, which hold model parameter values.

        :param variables: named and typed variables and values
        :type variables: dict
        """
        self._variables = {}
        self._momenta = {}
        
        self.update_variables(**variables)
        self.update_momenta(**momenta)

    @property
    def variables(self):
        """
        Returns a copy (!) of the variables and their values
        held by this state

        :returns: copy of variables
        :rtype: dict
        """
        return self._variables.copy()

    def update_variables(self, **variables):
        r"""
        Updates variables

        :param \**variables: named and typed variables to update
                             this state with
        :type \**variables: dict
        """
        self._variables.update(variables)
        
    @property
    def momenta(self):
        """
        I currently don't use this
        """
        return self._momenta.copy()
    
    def update_momenta(self, **momenta):
        """
        I currently don't use this
        """
        self._momenta.update(momenta)
