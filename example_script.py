"""
In this example, a polynomial is fitted to data points.
This serves to illustrate the interface of posterior components
and samplers you would need to implement for your applcation
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from binf.samplers import BinfState
from binf.example.samplers import make_sampler
from binf.example.plots import plot_fit, plot_hists
from binf.example.plots import plot_prediction_tube
from binf.example.misc import get_MAP, make_posterior
    
n_data_points = 20

real_coeffs = np.array([2.0, -4.0, 1.0, 1.5])
real_precision = 2.5
polynomial = np.polynomial.polynomial.polyval
xses = np.linspace(-2, 2, n_data_points)
ys = np.random.normal(loc=polynomial(xses, real_coeffs), 
                      scale=1.0 / np.sqrt(real_precision))

start = BinfState(dict(coefficients=np.ones(4), precision=1.0))

posterior = make_posterior(xses, ys, polynomial)

gips = make_sampler(posterior, 0.1, start)

samples = []
for i in range(30000):
    samples.append(deepcopy(gips.sample()))
    if i % 500 == 0 and i > 0:
        print("#### Gibbs sampling step {} ####".format(i)) 
        print('RWMC acceptance rate: {}'.format(gips.last_draw_stats['coefficients'].acceptance_rate))



samples_thin = samples[20000::20]
log_probs = np.array([posterior.log_prob(**x.variables) for x in samples_thin])

fig = plt.figure()
plot_hists(samples_thin, real_coeffs, real_precision, 30, fig)
fig.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plot_fit(xses, ys, polynomial, xses, log_probs, samples_thin, real_coeffs, real_precision, ax)
MAP_coeffs, _ = get_MAP(samples, log_probs)
MAP_fit = polynomial(xses, MAP_coeffs)
plot_prediction_tube(samples_thin, polynomial, xses, 
                     MAP_fit - 10, MAP_fit + 10, 150, ax)
                     
plt.show()
