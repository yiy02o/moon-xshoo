from check_broadeningMoonspectra import *


def flat_reflectance(wv, alb, alb_err, lims_to_fit, n_poly_flat=1, ax_toFit=None):
    if isinstance(lims_to_fit[0], list):
        # check if the limits are outside
        mask_nd = np.asarray([np.logical_and(wv.data > r[0], wv.data < r[1]) for r in lims_to_fit])
        wv_masked_nd = [np.ma.masked_array(wv.data, mask=~m) for m in mask_nd]
        al_masked_nd = [np.ma.masked_array(alb.data, mask=~m) for m in mask_nd]
        mask_max = np.asarray([np.ma.max(al_m) == alb.data for al_m in al_masked_nd])
        mask_maxInterval = [np.logical_or(~m_interval, ~m_max) for m_interval, m_max in zip(mask_nd, mask_max)]
        mask_region_ToFit = np.logical_and.reduce(mask_maxInterval)
        for r in lims_to_fit:
            ax_toFit.axvspan(r[0], r[1], alpha=.3, color="grey")
    else:
        mask_region_ToFit = np.logical_and(wv.data > lims_to_fit[0], wv.data < lims_to_fit[-1])
        ax_toFit.axvspan(lims_to_fit[0], lims_to_fit[1], alpha=.3, color="grey")
    m_poly = np.logical_or(wv.mask, mask_region_ToFit)
    wv_to_flat = np.ma.masked_array(wv.data, mask=m_poly)
    alb_to_flat = np.ma.masked_array(alb.data, mask=m_poly)
    # remove the nans
    final_mask = np.logical_or(np.isnan(alb_to_flat.data), wv_to_flat.mask)
    wv_to_flat = np.ma.masked_array(wv.data, mask=final_mask)
    alb_to_flat = np.ma.masked_array(alb.data, mask=final_mask)
    z = np.ma.polyfit(wv_to_flat, alb_to_flat, 1)
    # print(z)
    p = np.poly1d(z)
    oneD_poly = p(wv_to_flat.data)
    flat_al = np.ma.masked_array(alb.data / oneD_poly, mask=wv.mask)
    flat_al_err = np.ma.masked_array(alb_err.data / oneD_poly, mask=wv.mask)
    return flat_al, flat_al_err


def median_filter(x, k):
    assert k % 2 == 1
    assert x.ndim == 1
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.ma.median(y, axis=1)


def mgm(params, lamba):
    s1, s2, s3, mu1, mu2, mu3, sigma1, sigma2, sigma3 = params
    output = 1 + s1 * np.exp(-(lamba**(-1) - mu1**(-1))**2 / (2 * sigma1**2)) + \
             s2 * np.exp(-(lamba**(-1) - mu2**(-1))**2 / (2 * sigma2**2)) + \
             s3 * np.exp(-(lamba**(-1) - mu3**(-1))**2 / (2 * sigma3**2))
    return output


def minimize_function(lamba, s1, s2, s3, mu1, mu2, mu3, sigma1, sigma2, sigma3):
    params = s1, s2, s3, mu1, mu2, mu3, sigma1, sigma2, sigma3
    return mgm(params, lamba)


fig_general, ax_general = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
ALBFLAT = []
ALBFLATERR = []
# Corrected edge
for wv_global, alb_global, alb_br_global, alb_br_err_global, ax_g in zip(WVGLOBAL, ALBGLOBAL, ALBBROADGLOBAL,
                                                                         ALBBROADGLOBALERR, fig_general.axes):
    ax_g.grid()
    # ax_general[0].set_ylabel('Reflectance')
    # ax_general[0].set_xlabel(r'$\lambda$ (nm)')
    alb_flat, alb_flat_err = flat_reflectance(wv_global, alb_br_global, alb_br_err_global, [[1_550, 1_580], [720, 770]],
                                              n_poly_flat=1, ax_toFit=ax_g)
    ALBFLAT.append(alb_flat)
    ALBFLATERR.append(alb_flat_err)
    mask_band = np.logical_and(wv_global.data > 700, wv_global.data < 1_750)
    ax_g.plot(wv_global, alb_br_global)
    ax_g.plot(wv_global, alb_br_global / alb_flat, ls="--", c="r")
    ax_g.set_xlim(700, 1_750)
fig_general.show()

fig_fit, ax_fit = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
# fig_fit = plt.figure(layout='constrained', figsize=(15, 10))
subfigs_fit = fig_fit.subfigures(1, 2, wspace=0.01, width_ratios=[1, 1])
# Corrected edge
axs_loc = subfigs[1].subplots(2, 2)fig_fit.show()

#fig_err, ax_err = plt.subplots()
#for wv_global, err_global in zip(WVGLOBAL, ALBBROADGLOBALERR):
#    ax_err.plot(wv_global, err_global, lw=.5)
#fig_err.show()
