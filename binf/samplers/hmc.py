'''
HMC sampler
'''

from collections import namedtuple
from copy import deepcopy

import numpy as np

from csb.numeric import exp

HMCSampleStats = namedtuple('HMCSampleStats', 'accepted stepsize')


class HMCSampler(object):

    def __init__(self, pdf, state, timestep, nsteps, timestep_adaption_limit=0,
                 adaption_uprate=1.05, adaption_downrate=0.95, variable_name=None):

        self.pdf = pdf
        self.state = state
        self.timestep = timestep
        self.nsteps = nsteps
        self.timestep_adaption_limit = timestep_adaption_limit
        self.adaption_uprate = adaption_uprate
        self.adaption_downrate = adaption_downrate
        self._variable_name = variable_name

        self._last_move_accepted = 0
        self.counter = 0

    @property
    def variable_name(self):
        return 'HMC' if self._variable_name is None else self._variable_name

    @property
    def last_move_accepted(self):
        return self._last_move_accepted

    def _leapfrog(self, q, p, timestep, nsteps):

        gradient = lambda x: self.pdf.gradient(**{self._variable_name: x})
        
        p -= 0.5 * timestep * gradient(q)

        for i in range(nsteps-1):
            q += p * timestep
            p -= timestep * gradient(q)

        q += p * timestep
        p -= 0.5 * timestep * gradient(q)        

        return q, p

    def _copy_state(self, state):

        return deepcopy(state)

    def sample(self):

        V = lambda x: -self.pdf.log_prob(**{self._variable_name: x})

        q = self._copy_state(self.state)
        p = np.random.normal(size=q.shape)

        E_before = V(q) + 0.5 * np.sum(p ** 2)
        q, p = self._leapfrog(q, p, self.timestep, self.nsteps)
        E_after = V(q) + 0.5 * np.sum(p ** 2)
        acc = np.random.uniform() < exp(-(E_after - E_before))

        self._last_move_accepted = acc
        self.counter += 1

        if self.counter < self.timestep_adaption_limit:
            self._adapt_timestep()

        if acc:
            self._state = q
            return self._copy_state(q)
        else:
            return self._copy_state(self.state)

    @property
    def last_draw_stats(self):

        return {self.variable_name: HMCSampleStats(self.last_move_accepted,
                                                   self.timestep)}

    def _adapt_timestep(self):

        if self.last_move_accepted:
            self.timestep *= self.adaption_uprate
        else:
            self.timestep *= self.adaption_downrate
