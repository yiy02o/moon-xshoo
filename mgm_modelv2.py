from mgm_model import *
import emcee
import corner

############################################ MCMC #####################################################################


def single_model(params, data):
    s, mu, sigma = params
    x, y, yerr = data
    output = 1 + s * np.exp(-(x**(-1) - mu**(-1))**2 / (2*sigma**2))
    return output


def model(params, data):
    c0, c1, c2, s1, s2, s3, mu1, mu2, mu3, sigma1, sigma2, sigma3, Pb, Yb, logVb = params
    x, y, yerr = data
    c_lamba = np.log(c0 + c1 * x + c2 * x**2)
    output = c_lamba + s1 * np.exp(-(x**(-1) - mu1**(-1))**2 / (2*sigma1**2)) + \
             s2 * np.exp(-(x ** (-1) - mu2 ** (-1)) ** 2 / (2 * sigma2 ** 2)) + \
             s3 * np.exp(-(x ** (-1) - mu3 ** (-1)) ** 2 / (2 * sigma3 ** 2))
    return output


def lnlike(params, data):
    y_model = model(params, data)
    x, y, yerr = data
    log_y, log_yerr = np.log(y), np.fabs(yerr / y)
    c0, c1, c2, s1, s2, s3, mu1, mu2, mu3, sigma1, sigma2, sigma3, Pb, Yb, logVb = params
    Vb = 10**logVb
    pbb_good_point = (1 - Pb) * np.exp(-.5 * ((log_y - y_model) / log_yerr)**2) / np.sqrt(2 * np.pi * log_yerr**2)
    pbb_bad_point = Pb * np.exp(-.5 * ((y - Yb)**2 / (Vb + log_yerr**2))) / np.sqrt(2 * np.pi * (Vb + log_yerr**2))
    ln_like = np.sum(np.log(pbb_good_point + pbb_bad_point))
    return ln_like


def lnprior(params):
    c0, c1, c2, s1, s2, s3, mu1, mu2, mu3, sigma1, sigma2, sigma3, Pb, Yb, logVb = params
    if
        return 0.
    else:
        return -np.inf


def lnprob(params, data):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, data)


############################ SET UP #############################
nwalkers = 90
initial = np.array([-.02, -.1, -.02, 850, 1_050, 1_300, 1e-4, 1e-4, 1e-4])
ndim = len(initial)
niter = 10_000
p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]
# WVTOFITMARE[1].data[~WVTOFITMARE[1].mask]
# ALBFLATMARE[1].data[~ALBFLATMARE[1].mask]
# ALBFLATERRMARE[1].data[~ALBFLATERRMARE[1].mask]
DATASET = [[wv_arr.data[~wv_arr.mask] for wv_arr in WVTOFITMARE], [al_arr.data[~al_arr.mask] for al_arr in ALBFLATMARE],
           [al_err_arr.data[~al_err_arr.mask] for al_err_arr in ALBFLATERRMARE]]

########################### init ###############################################


def main(P0, NWALKERS, NITER, NDIM, LNPROB, data):
    sampler_main = emcee.EnsembleSampler(NWALKERS, NDIM, LNPROB, args=[data])

    # print("Running burn-in...")
    # p_0, _, _ = sampler.run_mcmc(P0, 100)
    # sampler.reset()

    print("Running production...")
    POS, PROB, STATE = sampler_main.run_mcmc(P0, NITER)

    return sampler_main, POS, PROB, STATE


############################ PLOT RESULTS ############################3#


def view_BP(sampler):
    bp_idx_s1, bp_idx_s2, bp_idx_s3 = 300_000, 300_000, 300_000
    bp_idx_mu1, bp_idx_mu2, bp_idx_mu3 = 300_000, 300_000, 300_000
    bp_idx_sigma1, bp_idx_sigma2, bp_idx_sigma3 = 300_000, 300_000, 300_000

    fig = plt.figure(layout='constrained', figsize=(35, 8))
    fig.suptitle("Check the parameter burning phase and distribution")
    subfigs = fig.subfigures(1, 3, wspace=0.05, width_ratios=[1, 1, 1])

    # parameter s_i
    subfigs[0].suptitle(r'$s$')
    axsLeft = subfigs[0].subplots(2, 3)
    # s1
    axs1 = axsLeft[0, 0]
    axs1.plot(sampler.flatchain[bp_idx_s1:, 0])

    axs4 = axsLeft[1, 0]
    axs4.hist(sampler.flatchain[bp_idx_s1:, 0], bins=100, density=True)
    axs4.set_xlabel(r'$s_{1}$')

    # s2
    axs2 = axsLeft[0, 1]
    axs2.plot(sampler.flatchain[bp_idx_s2:, 1])

    axs5 = axsLeft[1, 1]
    axs5.hist(sampler.flatchain[bp_idx_s2:, 1], bins=100, density=True)
    axs5.set_xlabel(r'$s_{2}$')

    # s3
    axs3 = axsLeft[0, 2]
    axs3.plot(sampler.flatchain[bp_idx_s3:, 2])

    axs6 = axsLeft[1, 2]
    axs6.hist(sampler.flatchain[bp_idx_s3:, 2], bins=100, density=True)
    axs6.set_xlabel(r'$s_{3}$')

    # parameter mu_i
    subfigs[1].suptitle(r'$\mu$')
    axsCenter = subfigs[1].subplots(2, 3)
    # mu1
    axmu1 = axsCenter[0, 0]
    axmu1.plot(sampler.flatchain[bp_idx_mu1:, 3])

    axmu4 = axsCenter[1, 0]
    axmu4.hist(sampler.flatchain[bp_idx_mu1:, 3], bins=100, density=True)
    axmu4.set_xlabel(r'$\mu_{1}$')

    # mu2
    axmu2 = axsCenter[0, 1]
    axmu2.plot(sampler.flatchain[bp_idx_mu2:, 4])

    axmu5 = axsCenter[1, 1]
    axmu5.hist(sampler.flatchain[bp_idx_mu2:, 4], bins=100, density=True)
    axmu5.set_xlabel(r'$\mu_{2}$')

    # mu3
    axmu3 = axsCenter[0, 2]
    axmu3.plot(sampler.flatchain[bp_idx_mu3:, 5])

    axmu6 = axsCenter[1, 2]
    axmu6.hist(sampler.flatchain[bp_idx_mu3:, 5], bins=100, density=True)
    axmu6.set_xlabel(r'$\mu_{3}$')

    # parameter sigma_i
    subfigs[2].suptitle(r'$\sigma$')
    axsRight = subfigs[2].subplots(2, 3)
    # sigma1
    axsigma1 = axsRight[0, 0]
    axsigma1.plot(sampler.flatchain[bp_idx_sigma1:, 6])

    axsigma4 = axsRight[1, 0]
    axsigma4.hist(sampler.flatchain[bp_idx_sigma1:, 6], bins=100, density=True)
    axsigma4.set_xlabel(r'$\sigma_{1}$')

    # sigma2
    axsigma2 = axsRight[0, 1]
    axsigma2.plot(sampler.flatchain[bp_idx_sigma2:, 7])

    axsigma5 = axsRight[1, 1]
    axsigma5.hist(sampler.flatchain[bp_idx_sigma2:, 7], bins=100, density=True)
    axsigma5.set_xlabel(r'$\sigma_{2}$')

    # sigma3
    axsigma3 = axsRight[0, 2]
    axsigma3.plot(sampler.flatchain[bp_idx_sigma3:, 8])

    axsigma6 = axsRight[1, 2]
    axsigma6.hist(sampler.flatchain[bp_idx_sigma3:, 8], bins=100, density=True)
    axsigma6.set_xlabel(r'$\sigma_{3}$')

    fig.show()


def plotter(sampler, dataset, idx):
    x, y, y_err = dataset
    fig_plotter, ax_plotter = plt.subplots(2, 1, sharex=True)
    samples = sampler.flatchain[300_000:]
    ax_plotter[0].plot(WVTOFITMARE[idx], ALBFLATMARE[idx], c="k")
    for params in samples[np.random.randint(len(samples), size=1_000)]:
        ax_plotter[0].plot(x, model(params, dataset), c="r", alpha=.1)
    ax_plotter[0].grid()
    ax_plotter[0].set_ylabel(r'Continuum removed')
    ax_plotter[1].plot(WVTOFITMARE[idx], ALBFLATERRMARE[idx])
    ax_plotter[1].grid()
    ax_plotter[1].set_xlabel(f"$\\lambda$ (nm)")
    ax_plotter[1].set_ylabel(r'$\sigma_{\lambda}$')
    fig_plotter.show()


def best_plotter(sampler, dataset, idx):
    samples = sampler.flatchain[300_000:]

    param_max = samples[np.argmax(sampler.flatlnprobability[300_000:])]
    print(param_max)
    best_fit_model = model(param_max, [WVTOFITMARE[idx].data, ALBFLATMARE[idx].data, ALBFLATERRMARE[idx].data])
    first_mgm = single_model(param_max[0::3], [WVTOFITMARE[idx].data, ALBFLATMARE[idx].data, ALBFLATERRMARE[idx].data])
    second_mgm = single_model(param_max[1::3], [WVTOFITMARE[idx].data, ALBFLATMARE[idx].data, ALBFLATERRMARE[idx].data])
    third_mgm = single_model(param_max[2::3], [WVTOFITMARE[idx].data, ALBFLATMARE[idx].data, ALBFLATERRMARE[idx].data])
    x, y, y_err = dataset
    fig_best, ax_best = plt.subplots(figsize=(10, 4))
    ax_best.plot(WVTOFITMARE[idx], ALBFLATMARE[idx], c="k", label="Data")
    ax_best.plot(WVTOFITMARE[idx].data, best_fit_model, c="r", label="Highest Likelihood Model")
    ax_best.plot(WVTOFITMARE[idx].data, first_mgm, label="1st MGM")
    ax_best.plot(WVTOFITMARE[idx].data, second_mgm, label="2nd MGM")
    ax_best.plot(WVTOFITMARE[idx].data, third_mgm, label="3rd MGM")
    ax_best.legend()
    ax_best.grid()
    ax_best.set_xlim(np.min(x), np.max(x))
    fig_best.show()
    return best_fit_model


def corner_plot(sampler):
    labels = [r'$s_{1}$', r'$s_{2}$', r'$s_{3}$', r'$\mu_{1}$', r'$\mu_{2}$', r'$\mu_{3}$',
              r'$\sigma_{1}$', r'$\sigma_{2}$', r'$\sigma_{3}$']
    samples = sampler.flatchain[300_000:]
    fig_corner = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[.16, .5, .84],
                               title_fmt=str('.5f'))
    fig_corner.show()


def sample_walkers(nsamples, flattened_chain, dataset, best_model, idx):
    x, y, y_err = dataset
    models = []
    draw = np.floor(np.random.uniform(0, len(flattened_chain), size=nsamples)).astype(int)
    params_set = flattened_chain[draw]
    for params in params_set:
        mod = model(params, dataset)
        models.append(mod)
    spread = np.std(models, axis=0)
    med_model = np.median(models, axis=0)
    fig_spread, ax_spread = plt.subplots(figsize=(10, 4))
    ax_spread.plot(WVTOFITMARE[idx], ALBFLATMARE[idx], c="k", label="Data")
    #ax_spread.plot(WVTOFITMARE[idx].data, best_model, c="r", label="Highest Likelihood Model")
    ax_spread.fill_between(x, med_model-spread, med_model+spread, color="grey", alpha=.5,
                           label=r'$1\sigma$ Posterior Spread')
    ax_spread.legend()
    ax_spread.grid()
    ax_spread.set_xlim(np.min(x), np.max(x))
    fig_spread.show()


for i, (wv_arr, al_arr, al_err_arr) in enumerate(zip(DATASET[0][1:-1], DATASET[1][1:-1], DATASET[2][1:-1])):
    print(f"OFFSET: {i}")
    dataset_i = wv_arr, al_arr, al_err_arr
    sampler_i, pos_i, prob_i, state_i = main(p0, nwalkers, niter, ndim, lnprob, dataset_i)
    view_BP(sampler_i)
    plotter(sampler_i, dataset_i, i)
    best_i = best_plotter(sampler_i, dataset_i, i)
    corner_plot(sampler_i)
    #sample_walkers(100, sampler_i.flatchain[300_000:], dataset_i, best_i, i)


