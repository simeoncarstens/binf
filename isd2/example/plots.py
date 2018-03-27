import numpy as np
import matplotlib.pyplot as plt
    
def plot_prediction_tube(samples, polynomial, predict_space, ys_from, ys_to, n_ys, ax):

    from isd2.example.misc import predict
    
    predicted_ys = np.array([np.linspace(ys_from[i], ys_to[i], n_ys)
                             for i, _ in enumerate(predict_space)])
    predicted_ys_probs = np.array([[predict(x, y, samples, polynomial) for y in predicted_ys[i]]
                                   for i, x in enumerate(predict_space)])
    cdfs = np.cumsum(predicted_ys_probs * (predicted_ys[:,1] - predicted_ys[:,0])[:,None], 1)
    lower_tube_lims = np.array([predicted_ys[i][np.where(cdfs[i] < 0.05)[0][-1]]
                                for i in range(len(predict_space))])
    upper_tube_lims = np.array([predicted_ys[i][np.where(cdfs[i] > 0.95)[0][0]]
                                for i in range(len(predict_space))])
    ax.plot(predict_space, lower_tube_lims, ls='--', c='r', 
            label='95% equal-tailed credible\ninterval for predictions')
    ax.plot(predict_space, upper_tube_lims, ls='--', c='r')
    ax.plot(predict_space, [np.trapz(predicted_ys[i] * predicted_ys_probs[i],
                                     predicted_ys[i])
                            for i in range(len(predict_space))], c='g',
            label='prediction')
    ax.legend()

def plot_fit(xses, ys, polynomial, predict_space, log_probs, samples, 
             real_coeffs, real_precision, ax):

    from isd2.example.misc import get_MAP
    
    MAP_coeffs, MAP_precision = get_MAP(samples, log_probs)

    ax.scatter(xses, ys)
    map_ys = polynomial(xses, MAP_coeffs)
    ax.plot(xses, map_ys, label='MAP estimate')        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
def plot_hists(samples, real_coeffs, real_precision, n_bins, fig):

    coeffs = np.array([x.variables['coefficients'] for x in samples])
    precisions = np.array([x.variables['precision'] for x in samples])

    for i in range(len(real_coeffs)):
        coeff_str = chr(65 + i)
        ax = fig.add_subplot(321 + i)
        ax.hist(coeffs[:,i], bins=n_bins, normed=True)
        ax.set_xlabel(coeff_str)
        ax.set_yticks(())
        ax.plot((real_coeffs[i], real_coeffs[i]), (0, ax.get_ylim()[-1]),
                ls='--', c='r', label='real {}'.format(coeff_str))
        ax.legend()

    ax = fig.add_subplot(321 + i + 1)
    ax.hist(precisions, bins=30, normed=True)
    ax.set_xlabel('precision')
    ax.set_yticks(())
    ax.plot((real_precision, real_precision), (0, ax.get_ylim()[-1]),
            ls='--', c='r', label='real precision')
    ax.legend()
