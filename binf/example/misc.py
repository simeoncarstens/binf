import numpy as np
    
def predict(x, y, samples, polynomial):

    from csb.numeric import log_sum_exp

    if True:
        f = lambda coefficients, precision, x=x, y=y: -0.5 * (polynomial(x, coefficients) - y) ** 2 * precision + 0.5 * np.log(precision) - 0.5 * np.log(2.0 * np.pi)
        integrands = np.array([f(**x.variables) for x in samples])
    else:
        from binf.example.likelihood import make_likelihood
        
        Lnew = make_likelihood(np.array([x]), np.array([y]), polynomial)
        integrands = np.array([Lnew.log_prob(**x.variables) for x in samples])

    return np.exp(log_sum_exp(integrands)) / len(samples)

def get_MAP(samples, log_probs):

    map_sample = samples[np.argmax(log_probs)]

    return map_sample.variables['coefficients'], map_sample.variables['precision']

def make_posterior(xses, ys, polynomial):

    from binf.pdf.posteriors import Posterior
    from binf.example.likelihood import make_likelihood
    from binf.example.priors import make_priors
    
    L = make_likelihood(xses, ys, polynomial)
    priors = make_priors()

    return Posterior({L.name: L}, priors)
