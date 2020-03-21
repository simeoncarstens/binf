import numpy as np
from collections import namedtuple

RWMCSampleStats = namedtuple('RWMCSampleStats', 'acceptance_rate')


class GammaSampler(object):
    
    def __init__(self, pdf, state):

        self.pdf = pdf
        self.state = state

    def _get_prior(self):

        from binf.example.priors import GammaPrior
        
        prior = filter(lambda p: 'precision' in p.variables,
                       list(self.pdf.priors.values()))[0]
        print(prior)
        if not isinstance(prior, GammaPrior):
            msg = 'Prior for precision is not a Gamma distribution'
            raise NotImplementedError(msg)
        
        return prior

    def _calculate_shape(self):

        prior = self._get_prior()
        n_data_points = len(self.pdf.likelihoods['points'].error_model.ys)
        
        return 0.5 * n_data_points + prior.shape - 1
        
    def _calculate_rate(self):

        args = dict(coefficients=self.pdf['coefficients'].value,
                    precision=1.0)
        r1 = -self.pdf.likelihoods['points'].log_prob(**args)
        r2 = self._get_prior().rate
        
        return r1 + r2
    
    def sample(self, state=42):

        rate = self._calculate_rate()        
        shape = self._calculate_shape()
        sample = np.random.gamma(shape) / rate

        self.state = sample
        
        return self.state


class RWMCSampler(object):

    def __init__(self, pdf, state, stepsize):

        self.pdf = pdf
        self.state = state 
        self.stepsize = stepsize
        self._n_moves = 0
        self._n_accepted_moves = 0

    @property
    def last_draw_stats(self):
        
        return {'coefficients': RWMCSampleStats(self.acceptance_rate)}

    @property
    def acceptance_rate(self):
        if self._n_moves > 0:
            return self._n_accepted_moves / float(self._n_moves) 
        else:
            return 0.0

    def sample(self):

        E_old = -self.pdf.log_prob(coefficients=self.state)
        change = np.random.uniform(low=-self.stepsize, high=self.stepsize,
                                   size=len(self.state))
        proposal = self.state + change
        E_new = -self.pdf.log_prob(coefficients=proposal)

        accepted = np.random.random() < np.exp(-(E_new - E_old))

        if accepted:
            self.state = proposal
            self._n_accepted_moves += 1

        self._n_moves += 1

        return self.state

def make_sampler(posterior, rwmc_stepsize, start_state):

    from binf.samplers.gibbs import GibbsSampler

    coeffs = start_state.variables['coefficients']
    precision = start_state.variables['precision']
    coefficients_pdf = posterior.conditional_factory(precision=precision)
    coefficients_sampler = RWMCSampler(coefficients_pdf,
                                       coeffs,
                                       rwmc_stepsize)

    precision_pdf = posterior.conditional_factory(coefficients=coeffs)
    precision_sampler = GammaSampler(precision_pdf, precision)

    subsamplers = {'coefficients': coefficients_sampler,
                   'precision': precision_sampler}

    return GibbsSampler(posterior, start_state, subsamplers)
