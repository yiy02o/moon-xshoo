from itertools import chain
from moon_after_molecfit import *
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline, interp1d
from PyAstronomy import pyasl
import astropy.units as u
import astropy.constants as C
from brokenaxes import brokenaxes


def lambda_masked(nd_arr, condition):
    mask = mask_tellSpikes(nd_arr, condition)
    masked_data = np.ma.masked_array(nd_arr, mask)
    return masked_data


def geometricAlbedo(E_nu, E_s_nu, slit_width, E_nu_err, E_s_err, d_S_B=0.9938*u.AU, solar_source='sun',
                    d_E_B=399_184.62*u.km):
    """This function returns the geometric albedo, most commonly known as apparent albedo or radiance factor or just
    reflectance spectrum. The formula is:
    A(lambda, alpha, i, e) = E_lunar(lambda, alpha, i, e)/Omega_lunar / E_sun(lambda)/pi
    where lambda, alpha, i and e are the wavelength, phase angle, incidence angle and emission angle respectively.
    E_lunar is the lunar irradiance measured by XSHOOTER, E_solar is the solar irradiance illuminating the Moon and
    Omega_lunar is the solid angle of the Moon's region observed from the observer's point of view."""
    d_solar = 1.*u.AU if solar_source == 'sun' else 70.*u.pc
    r_earth = 6378.14*u.km
    r_moon = 1737*u.km
    d_E_B_mean = 384_400*u.km
    a = slit_width*r_moon/900   # subtended Moon radius 900''
    b = 11*r_moon/900           # 11'' is the slit height of X-Shooter
    Omega = a*b/(d_E_B - r_earth)**2
    f_d = (d_S_B/d_solar)**2 * (d_E_B/d_E_B_mean)**2
    f_d = f_d.to(u.AU/u.AU)
    try:
        # data
        E_ratio = E_nu.data / E_s_nu.data
        output_data = E_ratio * f_d * np.pi / Omega
        output_data = np.ma.masked_array(output_data.value, mask=E_nu.mask)
        # error
        err_ratio_right = np.sqrt((E_nu_err.data/E_nu.data)**2 + (E_s_err.data/E_s_nu.data)**2)
        err_ratio = np.fabs(E_nu_err.data/E_s_err.data) * err_ratio_right
        output_err = err_ratio * f_d * np.pi / Omega
        output_err = np.ma.masked_array(output_err.value, mask=E_nu_err.mask)
    except:
        # data
        E_ratio = E_nu / E_s_nu
        output_data = E_ratio * f_d * np.pi / Omega  # / D_ak
        output_data = output_data.value
        # error
        err_ratio_right = np.sqrt((E_nu_err/E_nu)**2 + (E_s_err/E_s_nu)**2)
        err_ratio = np.fabs(E_nu_err/E_s_err) * err_ratio_right
        output_err = err_ratio * f_d * np.pi / Omega  # / D_ak
        output_err = output_err.value
    return output_data, output_err


def normalizeXXnm(y_arr, lamba, norm_range):
    mask_range = ~np.logical_and(lamba.data > norm_range[0], lamba.data < norm_range[1])
    mask_from_before = y_arr.mask
    new_mask = np.logical_or.reduce(np.asarray([mask_range, mask_from_before]))
    y_arr_masked = np.ma.masked_array(y_arr.data, mask=new_mask)
    median_to_norm = np.ma.median(y_arr_masked)
    return median_to_norm


def compute_dRV(w, f, tw, tf, rvmin, rvmax, drv, skipedge=20, plot=False):
    if plot:
        plt.clf()
        fig_rv, ax_rv = plt.subplots(3, 1, layout="constrained", figsize=(16, 6))
        ax_rv[0].set_title('Template (blue) and spectra shifted (red), both normalized, before RV correction')
        ax_rv[0].plot(tw, tf, 'b.-')
        ax_rv[0].plot(w, f, 'r.-')
        ax_rv[0].grid()

    rv, cc = pyasl.crosscorrRV(w, f, tw, tf, rvmin, rvmax, drv, skipedge=skipedge)
    maxind = np.argmax(cc)
    print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
    if rv[maxind] > 0:
        print(" A red-shift with respect to the template")
    else:
        print(" A blue-shift with respect to the template")
    if plot:
        ax_rv[1].plot(rv, cc, 'bp-')
        ax_rv[1].plot(rv[maxind], cc[maxind], 'ro')

        ax_rv[2].set_title('Template (blue) and spectra shifted (red), both normalized, after RV correction')
        ax_rv[2].plot(tw, tf, 'b.-')
        ax_rv[2].plot(w/(1 + rv[maxind]/C.c.to(u.km/u.s).value), f, 'r.-')
        ax_rv[2].grid()
        fig_rv.show()
    return rv[maxind]


def lambda_ranges_v2(Ek, lamba, n_poly, s):
    poly_k = norm_slope(Ek, lamba, n_poly)
    Ek_norm = Ek/poly_k
    Ek_clip = sigma_clip(Ek_norm, sigma=s, stdfunc=scipy.stats.iqr)
    m_clip = Ek_clip.mask
    lamba = np.ma.masked_array(lamba.data, mask=m_clip)
    return lamba


def fit_cont_v2(Ek, lamba, n_poly, s, n_poly_init=1, plot_cont=False, return_poly=False):
    lamba_clipped = lambda_ranges_v2(Ek, lamba, n_poly_init, s)
    m_clip = lamba_clipped.mask
    Ek_clipped = np.ma.masked_array(Ek.data, mask=m_clip)
    poly_k = norm_slope(Ek_clipped, lamba_clipped, n_poly)
    try:
        Ek_norm = Ek.data/poly_k
        Ek_norm = np.ma.masked_array(Ek_norm, mask=lamba.mask)
    except:
        Ek_norm = Ek/poly_k
    if plot_cont:
        plt.clf()
        fig_fitMedian, ax_fitMedian = plt.subplots(2, 1, layout="constrained")
        ax_fitMedian[0].set_title("Median region continuum fit and the polynomial")
        ax_fitMedian[0].plot(lamba, Ek, lw=0.5)
        ax_fitMedian[0].plot(lamba, Ek_clipped, c='k')
        ax_fitMedian[0].plot(lamba, poly_k, c='r')
        ax_fitMedian[0].set_xlabel(r'$\lambda$ (nm)', fontsize=14)

        ax_fitMedian[1].set_title("Median region continuum fit normalized")
        ax_fitMedian[1].plot(lamba, Ek_norm, lw=0.5)
        ax_fitMedian[1].set_ylabel('Normalized flux', fontsize=14)
        ax_fitMedian[1].set_xlabel(r'$\lambda$ (nm)', fontsize=14)
        fig_fitMedian.show()
    if return_poly:
        return poly_k
    return Ek_norm


def broad_moon_spectra(tw, tf, res_down):
    maxsigma = 5
    smoothed_tf_spec, fwhm = pyasl.instrBroadGaussFast(tw.data, tf.data, res_down, edgeHandling='firstlast',
                                                       fullout=True, maxsig=maxsigma)
    interp_func = interp1d(tw.data, smoothed_tf_spec, kind='slinear')
    spectra_smoothed = interp_func(tw.data)
    spectra_smoothed = np.ma.masked_array(spectra_smoothed, mask=tf.mask)
    return spectra_smoothed


def rv_and_broadening_iteration(tw, lamba_solar, tf, f_solar, f_solar_err, select,
                                param_guesses, rvmin, rvmax, drv, skipedge, best_solars, dlamba, rv=None,
                                initial_resolution=None, plot=False, ax_lims=None, best_method='mean',
                                propagate_convErr=False, corr_data=False, n_poly=1):
    """I have to write about this"""
    if rv is None:
        if select is None:
            raise ValueError(f"You should provide a wv range")
        if isinstance(select[0], (list, tuple)):
            idx_masked_tw = [np.where((tw.data > s[0]) & (tw.data < s[-1])) for s in select]
            idx_masked_w = [np.where((lamba_solar > s[0]) & (lamba_solar < s[-1])) for s in select]

            # cut the spectra and normalize
            tfNorm_torv_arr = [fit_cont_v2(tf.data[m], tw.data[m], 1, 1, plot_cont=True if id_tf == 0 else False)
                               for id_tf, m in enumerate(idx_masked_tw)]
            f_solarNorm_torv_arr = [fit_cont_v2(f_solar[m], lamba_solar[m], 1, 1,
                                                plot_cont=True if id_f == 0 else False)
                                    for id_f, m in enumerate(idx_masked_w)]
            f_norm_broad_test_arr = [pyasl.instrBroadGaussFast(lamba_solar[m], f_solar_norm,  # f_solar_norm_preBroad[m],
                                                               param_guesses[0], edgeHandling="firstlast",
                                                               fullout=True)[0] for m, f_solar_norm in \
                                     zip(idx_masked_w, f_solarNorm_torv_arr)]
            rv_per_chunk = [compute_dRV(lamba_solar[m_w], f_norm_broad, tw.data[m_tw], tf_norm,#tf_norm.data[m_tw],
                                        rvmin, rvmax,
                                        drv, skipedge=skipedge, plot=True if id_rv == 0 else False)
                            for id_rv, (m_tw, m_w, f_norm_broad, tf_norm) in \
                            enumerate(zip(idx_masked_tw, idx_masked_w, f_norm_broad_test_arr, tfNorm_torv_arr))]
            delta_per_chunk = [1 + rv_i / C.c.to(u.km / u.s).value for rv_i in rv_per_chunk]
            rv_chunks = [np.median(tw.data[m_tw]) for m_tw in idx_masked_tw]
            rv = np.median(rv_per_chunk)

    solar_wav_rvCorr = lamba_solar / (1 + rv / C.c.to(u.km / u.s).value)
    ################################################## ABSOLUTE SOLAR FLUX SHIFTED ####################################
    spl = UnivariateSpline(solar_wav_rvCorr, f_solar, s=0)
    solar_rvCorr = spl(tw)
    solar_rvCorr_w_rvCorr = spl(solar_wav_rvCorr)
    ################################# ABSOLUTE SOLAR FLUX UNC. SHIFTED (?) ############################################
    spl_err = UnivariateSpline(solar_wav_rvCorr, f_solar_err, s=0)
    solar_err_rvCorr = spl_err(tw)
    solar_err_rvCorr_w_rvCorr = spl_err(solar_wav_rvCorr)
    # if plot:
    if True:
        fig = plt.figure(layout='constrained', figsize=(18, 6))
        # fig.suptitle(f"Broadening process")
        subfigs = fig.subfigures(1, 3, wspace=0.05, width_ratios=[1, 1, 1])
        # subfigs[0].suptitle(f"Initial guesses $R = {np.round(param_guesses[0])}$ and "
        #                    f"$\sigma = {np.round(param_guesses[1], 2)}$")
        subfigs[0].suptitle(f"Initial guess $R* = {np.round(param_guesses[0]):.0f}$", fontsize=16)
        axsLeft = subfigs[0].subplots(3, 1)
        for idx, ax in enumerate(axsLeft):
            ax1 = ax.twinx()
            ax1.set_ylim(0.95, 1.05)
            mask_tw = np.logical_and(tw.data > ax_lims[idx][0], tw.data < ax_lims[idx][1])
            mask_solar_wav = np.logical_and(solar_wav_rvCorr > ax_lims[idx][0], solar_wav_rvCorr < ax_lims[idx][1])
            # normalize the three spectra that you want to plot
            tf_norm_toPlot = fit_cont_v2(tf.data[mask_tw], tw.data[mask_tw], 1, 1)
            solar_norm_rvCorr_tw_toPlot = fit_cont_v2(solar_rvCorr[mask_tw], tw.data[mask_tw], 1, 1)
            solar_norm_rvCorr_w_rvCorr_toPlot = fit_cont_v2(solar_rvCorr_w_rvCorr[mask_solar_wav],
                                                            solar_wav_rvCorr[mask_solar_wav], 1, 1)
            ax.plot(tw.data[mask_tw], tf_norm_toPlot, 'k.-')
            ax1.plot(tw.data[mask_tw], tf_norm_toPlot/solar_norm_rvCorr_tw_toPlot, 'b.-', alpha=0.7)
            ax.plot(solar_wav_rvCorr[mask_solar_wav], solar_norm_rvCorr_w_rvCorr_toPlot, c='g')
            if idx == 2:
                ax.set_xlabel(r'$\lambda$ (nm)', fontsize=15)

    def fitness_function(guesses, data_wav, data_flux, solar_wav, solar_flux_beforeBroad):
        resolution, sigma = guesses   # unpack parameters
        # Convolve the solar spectrum with the gaussian kernel
        # print(solar_flux_beforeBroad)
        smoothed_solar_spec, fwhm = pyasl.instrBroadGaussFast(solar_wav, solar_flux_beforeBroad, resolution,
                                                              edgeHandling='firstlast', fullout=True, maxsig=sigma)
        interpolate_func = interp1d(solar_wav, smoothed_solar_spec, kind='slinear')
        interp_smooth = interpolate_func(data_wav)
        # calculate the mean squared error
        mse = np.mean((interp_smooth - data_flux) ** 2)
        return mse
    if initial_resolution is None:
        initial_guess = np.array([param_guesses[0], param_guesses[1]])
        print(f"initial_guess", initial_guess)
        result_arr = []
        for i, line in enumerate(best_solars):
            # tf
            mask_tw_broad = np.abs(tw.data - line) <= dlamba
            tw_premasked = tw.data[mask_tw_broad]
            tf_norm_premasked = fit_cont_v2(tf.data[mask_tw_broad], tw_premasked, 1, 1)
            # solar
            mask_solar_broad = np.abs(solar_wav_rvCorr - line) <= dlamba
            solar_wav_rvCorr_masked = solar_wav_rvCorr[mask_solar_broad]
            solar_norm_rvCorr_w_rvCorr_masked = fit_cont_v2(solar_rvCorr_w_rvCorr[mask_solar_broad],
                                                            solar_wav_rvCorr_masked, 1, 1)

            tw_last_mask = (tw_premasked >= np.min(solar_wav_rvCorr_masked)) & \
                           (tw_premasked <= np.max(solar_wav_rvCorr_masked))
            tw_masked = tw_premasked[tw_last_mask]
            tf_norm_masked = tf_norm_premasked[tw_last_mask]
            result = minimize(fitness_function, initial_guess,
                              args=(tw_masked, tf_norm_masked, solar_wav_rvCorr_masked,
                                    solar_norm_rvCorr_w_rvCorr_masked),
                              method="Nelder-Mead")
            result_arr.append(result.x)
            if i % 3 == 0:
                print(line)
        result_arr = np.array(result_arr)
        if best_method not in ['mean', 'median', 'min']:
            raise ValueError("best method keyword invalid")
        best_resolution = np.mean(result_arr[:, 0]) if best_method == 'mean' else \
            np.median(result_arr[:, 0]) if best_method == 'median' else np.min(result_arr[:, 0])
        best_sigma = np.mean(result_arr[:, 1]) if best_method == 'mean' else \
            np.median(result_arr[:, 1]) if best_method == 'median' else \
                result_arr[:, 1][np.argmin(result_arr[:, 0])]
        print(f"Resolution, sigma:", best_resolution, best_sigma)
    if initial_resolution is not None:
        print(initial_resolution)
        best_resolution = initial_resolution
        best_sigma = 5

    # Now we use this optimal values to compute the smoothed solar spectrum
    ############################################### ABSOLUTE SOLAR FLUX CONVOLVED #####################################
    convolved_smooth_solarFlux, fwhm1 = pyasl.instrBroadGaussFast(solar_wav_rvCorr, solar_rvCorr_w_rvCorr,
                                                                  best_resolution, edgeHandling='firstlast',
                                                                  fullout=True, maxsig=best_sigma)
    interp_func = interp1d(solar_wav_rvCorr, convolved_smooth_solarFlux, kind='slinear', fill_value="extrapolate")
    smooth_solarFlux = interp_func(tw.data)

    ######################################## ABSOLUTE SOLAR FLUX UNC. CONVOLVED #######################################
    if propagate_convErr:
        meanWvl = np.mean(tw.data)
        if corr_data:
            # check the arm
            in_uvb = np.any(np.logical_and(tw.data > 420, tw.data < 430))
            in_vis = np.any(np.logical_and(tw.data > 650, tw.data < 652))
            in_nir = np.any(np.logical_and(tw.data > 2_000, tw.data < 2_002))
            # compute the original resolution of Solar spectra model
            res_original = 44_430 if in_uvb else 52_097 if in_vis else 44_475  # X-shooter manual
            fwhm_original = meanWvl / float(res_original)
            sigma_original = fwhm_original / (2. * np.sqrt(2. * np.log(2.)))
            print(f"Original kernel width: {sigma_original:.5f}")
            fwhm_kernel = meanWvl / float(best_resolution)
            sigma_kernel = fwhm_kernel / (2. * np.sqrt(2. * np.log(2.)))
            print(f"Kernel width from the flux convolution process: {sigma_kernel:.5f}")
            solar_flux_variance = solar_err_rvCorr_w_rvCorr ** 2
            convolved_solarFlux_variance = pyasl.broadGaussFast(solar_wav_rvCorr, solar_flux_variance,
                                                                sigma_kernel, edgeHandling='firstlast',
                                                                maxsig=best_sigma, square=True)
            convolved_solarFlux_variance_gaussProp = pyasl.broadGaussFast(solar_wav_rvCorr, solar_flux_variance,
                                                                          sigma_kernel, edgeHandling='firstlast',
                                                                          maxsig=best_sigma, square=False)
            scale_factor = 2*np.sqrt(np.pi)*sigma_original*sigma_kernel / np.sqrt(sigma_original**2 + sigma_kernel**2)
            print(scale_factor)
            convolved_solarFlux_err = np.sqrt(scale_factor * convolved_solarFlux_variance)
            convolved_solarFlux_err_gaussProp = np.sqrt(convolved_solarFlux_variance_gaussProp)
            interp_solarFlux_err = interp1d(solar_wav_rvCorr, convolved_solarFlux_err, kind='slinear',
                                            fill_value="extrapolate")
            interp_solarFlux_err_gaussProp = interp1d(solar_wav_rvCorr, convolved_solarFlux_err_gaussProp,
                                                      kind='slinear', fill_value="extrapolate")
            smooth_solarFlux_err = interp_solarFlux_err(tw.data)
            smooth_solarFlux_err_gaussProp = interp_solarFlux_err_gaussProp(tw.data)
        else:
            fwhm_kernel = meanWvl / float(best_resolution)
            sigma_kernel = fwhm_kernel / (2. * np.sqrt(2. * np.log(2.)))
            print(f"Kernel width from the flux convolution process: {sigma_kernel}")
            solar_flux_variance = solar_err_rvCorr_w_rvCorr**2
            convolved_solarFlux_variance = pyasl.broadGaussFast(solar_wav_rvCorr, solar_flux_variance,
                                                                sigma_kernel, edgeHandling='firstlast',
                                                                maxsig=best_sigma, square=True)
            convolved_solarFlux_variance_gaussProp = pyasl.broadGaussFast(solar_wav_rvCorr, solar_flux_variance,
                                                                          sigma_kernel, edgeHandling='firstlast',
                                                                          maxsig=best_sigma, square=False)
            convolved_solarFlux_err = np.sqrt(convolved_solarFlux_variance)
            convolved_solarFlux_err_gaussProp = np.sqrt(convolved_solarFlux_variance_gaussProp)
            interp_solarFlux_err = interp1d(solar_wav_rvCorr, convolved_solarFlux_err, kind='slinear',
                                            fill_value="extrapolate")
            interp_solarFlux_err_gaussProp = interp1d(solar_wav_rvCorr, convolved_solarFlux_err_gaussProp,
                                                      kind='slinear', fill_value="extrapolate")
            smooth_solarFlux_err = interp_solarFlux_err(tw.data)
            smooth_solarFlux_err_gaussProp = interp_solarFlux_err_gaussProp(tw.data)

    #if plot:
    if True:
        # Distribution of the parameters
        # axsCenter = subfigs[1].subplots(3, 1)
        axsCenter = subfigs[1].subplots(2, 1)
        if initial_resolution is None:
            axsCenter[0].hist(result_arr[:, 0])
            axsCenter[0].set_xlabel("R*", fontsize=15)
            axsCenter[0].axvline(best_resolution, c="k")
            # axsCenter[1].plot(best_solars, result_arr[:, 0], label=r'$R_{\lambda}$', c="b")
            axsCenter[1].plot(best_solars, result_arr[:, 0], c="b")
            axsCenter[1].set_xlabel(r'$\lambda$ (nm)', fontsize=15)
            axsCenter[1].set_ylabel(f"$R*$", fontsize=15)
            axsCenter[1].axhline(best_resolution, c="k", ls="--")
            # ax_sigma = axsCenter[1].twinx()
            # ax_sigma.plot(best_solars, result_arr[:, 1], label=r'$\sigma_{\lambda}$', c="m")
            # ax_sigma.set_ylabel(f"$\\sigma$")
            # ax_sigma.axhline(best_sigma, c="grey", ls="--")
            # axsCenter[1].legend(fontsize=7)
            # ax_sigma.legend(fontsize=7)
            # axsCenter[2].hist(result_arr[:, 1])
            # axsCenter[2].set_xlabel(f"$\\sigma$")
            # axsCenter[2].axvline(best_sigma, c="k", ls="--")
        # example of the broadened spectra
        # subfigs[2].suptitle(f"Best parameters $R = {np.round(best_resolution)}$ and "
        #                    f"$\\sigma = {np.round(best_sigma, 2)}$")
        subfigs[2].suptitle(f"Best parameter $R = {np.round(best_resolution):.0f}$", fontsize=16)
        axsRight = subfigs[2].subplots(3, 1)
        for idx, ax in enumerate(axsRight):
            ax1 = ax.twinx()
            ax1.set_ylim(0.95, 1.05)
            mask_tw = np.logical_and(tw.data > ax_lims[idx][0], tw.data < ax_lims[idx][1])
            # mask_solar_wav = np.logical_and(solar_wav_rvCorr > ax_lims[idx][0], solar_wav_rvCorr < ax_lims[idx][1])
            # normalize the three spectra that you want to plot
            tf_norm_toPlot = fit_cont_v2(tf.data[mask_tw], tw.data[mask_tw], 1, 1)
            smoothSolar_norm_toPlot = fit_cont_v2(smooth_solarFlux[mask_tw], tw.data[mask_tw], 1, 1)
            ax.plot(tw.data[mask_tw], tf_norm_toPlot, 'k.-')
            ax1.plot(tw.data[mask_tw], tf_norm_toPlot/smoothSolar_norm_toPlot, 'b.-', alpha=0.7)
            ax.plot(tw.data[mask_tw], smoothSolar_norm_toPlot, c='g')
            if idx == 2:
                ax.set_xlabel(r'$\lambda$ (nm)', fontsize=15)
        fig.show()
        if propagate_convErr:
            scale_plot = 7 if not corr_data else 50
            fig_br, ax_br = plt.subplots(3, 1, figsize=(15, 6))
            for idx, ax in enumerate(ax_br):
                m_solar_err = np.logical_and(solar_wav_rvCorr > ax_lims[idx][0], solar_wav_rvCorr < ax_lims[idx][1])
                m_solar_br_err = np.logical_and(tw.data > ax_lims[idx][0], tw.data < ax_lims[idx][1])
                ax.plot(solar_wav_rvCorr[m_solar_err], f_solar_err[m_solar_err], 'r.-',
                        label=r'$\sigma$ original: ' + r'$\sqrt{Var(O_{x})}$')
                ax.plot(tw.data[m_solar_br_err], smooth_solarFlux_err_gaussProp[m_solar_br_err], 'b.-',
                        label='$\sigma$ only convolved-down')
                ax.plot(tw.data[m_solar_br_err], scale_plot * smooth_solarFlux_err[m_solar_br_err], 'k.-',
                        label='$\sigma$ propagated' + f" ({scale_plot}x):" + r'$\sqrt{Var(C_{x})}$')
            ax_br[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(.23, 1.25), fancybox=True, shadow=True,
                            ncol=3)
            ax_br[1].set_ylabel(r'Error ($\sigma_{\lambda}$) ' + r'W $m^{-2}$ nm$^{-1}$')
            ax_br[2].set_xlabel(r'$\lambda$ (nm)')
            title_var = r'Uncorrelated data method, using ' \
                        r'Var$(C_{x}) = \sum_{\vec z} K_{b}^{2}(\vec x - \vec z) \sigma_{\vec z}^{2}$, ' \
                        f"where $b={sigma_kernel:.3f}$ " \
                        f"with $R \sim {(sigma_kernel*(2.*np.sqrt(2.*np.log(2.)))/meanWvl)**-1:.0f}$" if not corr_data \
                else r'Correlated data method, using ' \
                     r'Var$(C_{x}) = \frac{2\sqrt{\pi}\theta b}{\sqrt{\theta^{2} + b^2}}' \
                     r'\sum_{\vec \delta} K_{\theta}^{2}(\vec \delta - \vec z) \sigma_{\vec \delta }^{2}$, ' \
                     f"where $b={sigma_original:.3f}$, " \
                     f"$R_b \sim {(sigma_original*(2.*np.sqrt(2.*np.log(2.)))/meanWvl)**-1:.0f}$, " \
                     f"$\\theta={sigma_kernel:.3f}$"
            fig_br.suptitle(title_var)
            fig_br.show()
    return rv, [best_resolution, best_sigma], [smooth_solarFlux, smooth_solarFlux_err]


def solar_smoothness(tw, tf, select, best_solar, tf_err, dlamba=10, drv=0.1, rvmin=-100, rvmax=100, rv=None,
                     skipedge=800, plot=False, best_method='mean', initial_resolution=None, propagate_convErr=False,
                     corr_data=False, **kwargs):
    path_hybridsolar005nm = '/home/yiyo/Downloads/solarhybrid005nm.csv'
    hybridsolar005nm = np.loadtxt(path_hybridsolar005nm, skiprows=1, delimiter=',')
    in_uvb = np.any(np.logical_and(tw.data > 420, tw.data < 430))
    in_vis = np.any(np.logical_and(tw.data > 650, tw.data < 652))
    in_nir = np.any(np.logical_and(tw.data > 2_000, tw.data < 2_002))
    if not np.any([in_uvb, in_vis, in_nir]):
        raise ValueError("Template wavelength not covered")
    ############# THIS CAN BE MORE EXPLICIT ##########################
    lamba_solar = hybridsolar005nm.T[0][:300_000] if in_uvb else hybridsolar005nm.T[0][240_000:750_000] if in_vis \
        else hybridsolar005nm.T[0][740_000:2_150_000]
    flux_solar = hybridsolar005nm.T[1][:300_000] if in_uvb else hybridsolar005nm.T[1][240_000:750_000] if in_vis \
        else hybridsolar005nm.T[1][740_000:2_150_000]
    flux_solar_err = hybridsolar005nm.T[2][:300_000] if in_uvb else hybridsolar005nm.T[2][240_000:750_000] if in_vis \
        else hybridsolar005nm.T[2][740_000:2_150_000]
    # n_poly = 5 if in_uvb else 7 if in_vis else 7
    resolution_guess = 9_700 if in_uvb else 18_400 if in_vis else 11_600
    slit_width = 0.5 if in_uvb else 0.4
    ax_lims = [[406, 409], [476, 479], [501, 504]] if in_uvb else [[588, 591], [746, 749], [806, 809]] if in_vis \
        else [[1_240, 1_245], [1_590, 1_595], [2_100, 2_105]]
    # First evaluation
    rv_0, best_params_0, smooth_solar_fluxes_0 = rv_and_broadening_iteration(tw, lamba_solar, tf, flux_solar,
                                                                             flux_solar_err, select,
                                                                             [resolution_guess, 1], rvmin, rvmax, drv,
                                                                             skipedge, best_solar, dlamba, rv=rv,
                                                                             plot=plot, ax_lims=ax_lims,
                                                                             best_method=best_method,
                                                                             initial_resolution=initial_resolution,
                                                                             propagate_convErr=propagate_convErr,
                                                                             corr_data=corr_data,)
    # It gives back in u.W / u.m**2 / u.nm).to(u.erg / u.s / u.cm**2 / u.AA) units
    ################################################ ABSOLUTE SOLAR FLUX #############################################
    smooth_flux = (smooth_solar_fluxes_0[0] * u.W / u.m**2 / u.nm).to(u.erg / u.s / u.cm**2 / u.AA)
    smooth_flux = np.ma.masked_array(smooth_flux.value, mask=tf.mask)
    ################################################ ABSOLUTE FLUX UNC. ##############################################
    smooth_flux_err = (smooth_solar_fluxes_0[1] * u.W / u.m ** 2 / u.nm).to(u.erg / u.s / u.cm ** 2 / u.AA)
    smooth_flux_err = np.ma.masked_array(smooth_flux_err.value, mask=tf.mask)

    ################### I have to improve the propagation of convolution error ##############################
    A_g, A_g_err = geometricAlbedo(tf, smooth_flux, slit_width, tf_err, smooth_flux_err, solar_source='sun', **kwargs)
    # print(f"Telluric zone in albedo: {A_g.data[A_g.mask][100:120]}")
    return rv_0, A_g, smooth_flux, A_g_err, smooth_flux_err


def oneForAll(tw, tf, tf_err, w=None, f=None, f_err=None, solar_source='sun', rv=None, n_sigma=1, select=None,
              best_solar=None, drv=0.1, rvmin=-50, rvmax=50, skipedge=20, dlamba=10, plot=False, best_method='mean',
              initial_resolution=None, propagate_convErr=False, corr_data=False, **kwargs):
    """I have to re-write this """

    if solar_source not in ['sun', 'solar_twin', 'earthshine', 'comparing_moon_zones']:
        raise ValueError("solar source keyword invalid")
    # In the case of a non sun correction, check if w and f have been submitted
    if solar_source != 'sun' and np.logical_or(w is None, f is None):
        raise ValueError("You should introduce lambda (w) and flux (f) quantities as inputs")
    in_uvb = np.any(np.logical_and(tw.data > 422., tw.data < 430.))
    slit_width = 0.5 if in_uvb else 0.4
    if solar_source == 'sun':
        return solar_smoothness(tw, tf, select, best_solar, tf_err, dlamba=dlamba, drv=drv, rvmin=rvmin, rvmax=rvmax,
                                skipedge=skipedge, plot=plot, best_method=best_method,
                                initial_resolution=initial_resolution, propagate_convErr=propagate_convErr,
                                corr_data=corr_data, **kwargs)
    ################################### I have to check for solar-star the uncertainties ###########################
    n_poly = 4 if in_uvb else 6
    f_normalized = fit_cont_v2(f, w, n_poly, n_sigma, plot_cont=plot)
    tf_normalized = fit_cont_v2(tf, tw, n_poly, n_sigma, plot_cont=plot)

    # RV compute
    if rv is None:
        if select is None:
            raise ValueError(f"You should provide a wv range")
        if isinstance(select[0], (list, tuple)):
            idx_masked = [np.where((tw.data > s[0]) & (tw.data < s[-1])) for s in select]
            rv = np.mean([compute_dRV(w.data[m], f_normalized.data[m], tw.data[m], tf_normalized.data[m], rvmin, rvmax,
                                      drv, skipedge=skipedge, plot=plot) for m in idx_masked])
    print(f"The mean rv is {rv}")
    w_corr = tw.data/(1 + rv/C.c.to(u.km/u.s).value)
    ########################################### DATA ########################################################
    spl = UnivariateSpline(w_corr, f.data, s=0)
    data_interp = spl(tw)
    f_interp = np.ma.masked_array(data_interp, mask=tf.mask)
    ########################################### UNC. ########################################################
    spl_err = UnivariateSpline(w_corr, f_err.data, s=0)
    f_interp_err = spl_err(tw)
    f_interp_err = np.ma.masked_array(f_interp_err, mask=tf.mask)
    ########################################### DATA NORMALIZED ##############################################
    spl_norm = UnivariateSpline(w_corr, f_normalized.data, s=0)
    data_norm_interp = spl_norm(tw)
    f_normalized_interp = np.ma.masked_array(data_norm_interp, mask=tf.mask)
    if plot:
        fig_pl, ax_pl = plt.subplots()
        ax_pl.plot(tw[5900:6000], tf_normalized[5900:6000], 'b.-', label='Template data')
        ax_pl.plot(w_corr[5900:6000], f_normalized_interp[5900:6000], 'r.-', label='Interpolated data')
        ax_pl.legend()
        ax_pl.grid()
        ax_pl.set_xlabel(f"$\lambda$ (nm)")
        ax_pl.set_ylabel(f"Normalized flux")
        fig_pl.show()
    A_g, A_g_err = geometricAlbedo(tf, f_interp, slit_width, tf_err, f_interp_err, solar_source='solar_twin', **kwargs)
    return rv, A_g, f_interp, A_g_err, f_interp_err


def plot_global_reflectance(lamba_arr, data_arr, select_rv_arm_arr, data_err_arr, lamba1_arr=None, data1_arr=None,
                            data1_err_arr=None, solar_source='sun', best_solar_arm_arr=None, dlamba=20, x_lims=None,
                            more_offsets=False, plot_corrections=False, best_method='mean', initial_resolution=None,
                            propagate_convErr=False, corr_data=False, **kwargs):
    A_g_data = []
    A_g_err = []
    rv = []
    f_interp_data = []
    f_interp_data_err = []
    lamba_data = []
    lamba_mask = []
    if solar_source == 'sun':
        if len(lamba_arr) == 3:
            lamba_arr = [lamba_arr[0], lamba_arr[1], lamba_arr[2][:22_000]]
            data_arr = [data_arr[0], data_arr[1], data_arr[2][:22_000]]
            if data_err_arr is not None and len(data_err_arr) == 3:
                data_err_arr = [data_err_arr[0], data_err_arr[1], data_err_arr[2][:22_000]]
        for idx, (lamba_arm, data_arm, select_rv_arm, best_solar_arm, data_err_arm) in enumerate(zip(lamba_arr,
                                                                                                     data_arr,
                                                                                                     select_rv_arm_arr,
                                                                                                     best_solar_arm_arr,
                                                                                                     data_err_arr)):
            rv_arm, A_g_arm, f_int_arm, A_g_err_arm, \
                f_int_err_arm = oneForAll(lamba_arm, data_arm, data_err_arm, solar_source=solar_source,
                                          select=select_rv_arm, best_solar=best_solar_arm, drv=0.1, rvmin=-100,
                                          rvmax=100, skipedge=800, dlamba=dlamba, plot=plot_corrections,
                                          best_method=best_method, initial_resolution=initial_resolution,
                                          propagate_convErr=propagate_convErr, corr_data=corr_data, **kwargs)
            mask_arm = np.zeros(len(lamba_arm.data), dtype=bool) if isinstance(lamba_arm.mask, np.bool_) else \
                lamba_arm.mask
            A_g_data.append(A_g_arm.data)
            A_g_err.append(A_g_err_arm.data)
            rv.append(rv_arm)
            lamba_data.append(lamba_arm.data)
            lamba_mask.append(mask_arm)
            f_interp_data.append(f_int_arm)
            f_interp_data_err.append(f_int_err_arm)
            print('Estoy en loop de oneForAll')
    if solar_source == 'solar_twin':
        for idx, (lamba_arm, data_arm, lamba1_arm, data1_arm, select_rv_arm, data_err_arm, data1_err_arm) in \
                enumerate(zip(lamba_arr, data_arr, lamba1_arr, data1_arr, select_rv_arm_arr, data_err_arr,
                              data1_err_arr)):
            rv_arm, A_g_arm, f_int_arm, A_g_err_arm, \
                f_int_err_arm = oneForAll(lamba_arm, data_arm, data_err_arm, w=lamba1_arm, f=data1_arm,
                                          f_err=data1_err_arm, solar_source=solar_source, select=select_rv_arm,
                                          plot=plot_corrections, **kwargs)
            mask_arm = np.zeros(len(lamba_arm.data), dtype=bool) if isinstance(lamba_arm.mask, np.bool_) else \
                lamba_arm.mask
            A_g_data.append(A_g_arm.data)
            A_g_err.append(A_g_err_arm.data)
            rv.append(rv_arm)
            lamba_data.append(lamba_arm.data)
            lamba_mask.append(mask_arm)
            f_interp_data.append(f_int_arm)
            f_interp_data_err.append(f_int_err_arm)
            print('Estoy en loop de oneForAll')
    print('Salí del print de OneForAll')
    lamba = np.ma.masked_array(list(chain.from_iterable(lamba_data)), mask=list(chain.from_iterable(lamba_mask)))
    A_g = np.ma.masked_array(list(chain.from_iterable(A_g_data)), mask=list(chain.from_iterable(lamba_mask)))
    A_g_err = np.ma.masked_array(list(chain.from_iterable(A_g_err)), mask=list(chain.from_iterable(lamba_mask)))
    if x_lims is not None:
        mask_now = np.logical_or(lamba.mask, ~np.logical_and(lamba.data > x_lims[0], lamba.data < x_lims[1]))
        lamba = np.ma.masked_array(lamba.data, mask=mask_now)
        A_g = np.ma.masked_array(A_g.data, mask=mask_now)
        A_g_err = np.ma.masked_array(A_g_err.data, mask=mask_now)
    if more_offsets:
        return lamba, A_g, f_interp_data[0], A_g_err, f_interp_data_err[0]
    else:
        pass


def sortSolarTwin(plot=False):
    path_SolarType_UVB = '/home/yiyo/moon_xshoo_pre_molecfit/reflexData_stdTell_solarType/reflex_end_products/' \
                         '2023-08-17T23:54:12/first/XSHOO.2018-03-10T10:02:47.844_tpl/' \
                         'Hip098197_TELL_SLIT_FLUX_MERGE1D_UVB.fits'
    path_SolarType_IDP_UVB = '/home/yiyo/moon_xshoo_pre_molecfit/reflexData_stdTell_solarType/reflex_end_products/' \
                             '2023-08-17T23:54:12/first/XSHOO.2018-03-10T10:02:47.844_tpl/' \
                             'Hip098197_TELL_SLIT_FLUX_IDP_UVB.fits'
    path_SolarType_VIS = '/home/yiyo/moon_xshoo_after_molecfit/stdTell_solarType/reflex_end_products/molecfit/XSHOOTER/' \
                         '2023-08-18T02:19:22/Hip098197_TELL_SLIT_FLUX_MERGE1D_VIS/Hip098197_SCIENCE_TELLURIC_CORR.fits'
    path_SolarType_NIR = '/home/yiyo/moon_xshoo_after_molecfit/stdTell_solarType/reflex_end_products/molecfit/XSHOOTER/' \
                         '2023-08-18T02:19:22/Hip098197_TELL_SLIT_FLUX_MERGE1D_NIR/Hip098197_SCIENCE_TELLURIC_CORR.fits'
    ################################################## UVB ##########################################################
    hdul_uvb_idp = fits.open(path_SolarType_IDP_UVB)
    wvsolar_uvb = hdul_uvb_idp[1].data[0]['WAVE'].copy()
    hdul_uvb_idp.close()
    hdul_uvb = fits.open(path_SolarType_UVB)
    js_uvb = hdul_uvb[0].data.copy()
    jserr_uvb = hdul_uvb[1].data.copy()
    hdul_uvb.close()
    crop_uvb = np.where(np.logical_and(wvsolar_uvb > 374, wvsolar_uvb < 3_000))
    wvsolar_uvb = wvsolar_uvb[crop_uvb]
    js_uvb = js_uvb[crop_uvb]
    jserr_uvb = jserr_uvb[crop_uvb]
    wvsolar_uvb = lambda_masked(wvsolar_uvb, mask_all)
    js_uvb = np.ma.masked_array(js_uvb, wvsolar_uvb.mask)
    jserr_uvb = np.ma.masked_array(jserr_uvb, wvsolar_uvb.mask)
    ################################################## VIS ##########################################################
    hdul_vis = fits.open(path_SolarType_VIS)
    wvsolar_vis = hdul_vis[4].data['lambda'].copy() * u.micron.to(u.nm)
    js_vis = hdul_vis[0].data.copy()
    jserr_vis = hdul_vis[1].data.copy()
    hdul_vis.close()
    crop_vis = np.where(np.logical_and(wvsolar_vis > 554, wvsolar_vis < 3_000))
    wvsolar_vis = wvsolar_vis[crop_vis]
    js_vis = js_vis[crop_vis]
    jserr_vis = jserr_vis[crop_vis]
    wvsolar_vis = lambda_masked(wvsolar_vis, mask_all)
    js_vis = np.ma.masked_array(js_vis, wvsolar_vis.mask)
    jserr_vis = np.ma.masked_array(jserr_vis, wvsolar_vis.mask)
    ################################################## NIR ##########################################################
    hdul_nir = fits.open(path_SolarType_NIR)
    wvsolar_nir = hdul_nir[4].data['lambda'].copy() * u.micron.to(u.nm)
    js_nir = hdul_nir[0].data.copy()
    jserr_nir = hdul_nir[1].data.copy()
    hdul_nir.close()
    crop_nir = np.where(np.logical_and(wvsolar_nir > 1050, wvsolar_nir < 3_000))
    wvsolar_nir = wvsolar_nir[crop_nir]
    js_nir = js_nir[crop_nir]
    jserr_nir = jserr_nir[crop_nir]
    wvsolar_nir = lambda_masked(wvsolar_nir, mask_all)
    js_nir = np.ma.masked_array(js_nir, wvsolar_nir.mask)
    jserr_nir = np.ma.masked_array(jserr_nir, wvsolar_nir.mask)

    wv_list = [wvsolar_uvb, wvsolar_vis, wvsolar_nir]
    js_list = [js_uvb, js_vis, js_nir]
    jserr_list = [jserr_uvb, jserr_vis, jserr_nir]
    wv_nd = np.ma.concatenate((wvsolar_uvb, wvsolar_vis, wvsolar_nir))
    js_nd = np.ma.concatenate((js_uvb, js_vis, js_nir))
    jserr_nd = np.ma.concatenate((jserr_uvb, jserr_vis, jserr_nir))
    if plot:
        fig_js, ax_js = plt.subplots()
        ax_js.plot(wv_nd, js_nd, lw=.5)
        ax_errjs = ax_js.twinx()
        ax_errjs.plot(wv_nd, jserr_nd, lw=.5, c="r", alpha=.8)
        fig_js.show()
    return wv_list, js_list, jserr_list


color_regions = ['cyan', 'b', 'orange', 'y', 'm']
alpha_0 = 100
rv_star = -28.3
npoly_per_region_per_arm = [[4, 6, 0], [4, 6, 0], [5, 6, 6], [5, 6, 6], [4, 6, 4]]
rangeToNorm = [745, 755]
zoomx = [550, 560]
select_rv_UVB = [[390, 425], [430, 460], [480, 520], [525, 540]]
select_rv_VIS = [[580, 600], [610, 620], [638, 660], [730, 750], [790, 812], [825, 890]]
select_rv_NIR = [[1090, 1100], [1180, 1230], [1500, 1700]]  # [1280, 1320], [1500, 1700]]
select_rv = [select_rv_UVB, select_rv_VIS, select_rv_NIR]
# BrokenXaxis
# lamba rest of the transition of interest
lambarestVIS = [585.37, 588.38, 588.99, 589.28, 589.59, 591.42, 593.01, 656.30]

# xlims range
selectUVB = tuple([(392.9, 393.9), (396.3, 397.3), (485.6, 486.6), (526.5, 527.5), (541.9, 542.9)])
selectVIS = tuple([(584.8, 585.8), (587.8, 588.8), (588.85, 589.85), (849, 851), (1049, 1051)])
selectNIR = tuple([(1240, 1241), (1245, 1246), (1499, 1500), (1555, 1556), (1559.5, 1560.5), (2200, 2201)])

# Convolve-down range
best_solar_UVB = [405, 415, 425, 435, 445, 455, 465, 475, 485, 495, 505, 510, 515, 525, 535, 530, 540, 545]
best_solar_VIS = [570, 590, 601, 656, 735, 745, 800, 840, 855, 865, 880]
best_solar_NIR = [1_185, 1_190, 1_200, 1_210, 1_220, 1_230, 1_300, 1_310, 1_510, 1_520, 1_530, 1_540, 1_550, 1_560,
                  1_570, 1_580, 1_590, 1_600, 1_610, 1_620, 1_630, 1_640, 1_650, 1_660, 1_670, 1_680, 1_690, 2_060,
                  2_070, 2_080, 2_090, 2_100, 2_110, 2_120, 2_130, 2_140, 2_150, 2_160, 2_170, 2_180, 2_190, 2_200,
                  2_210, 2_220, 2_230, 2_320, 2_330]
best_solar_all = [best_solar_UVB, best_solar_VIS, best_solar_NIR]

# Photometric latitudes and longitudes of each region
################################################ Mare Imbrium (HL in the fits) ########################################
hl1 = [-26.97, 37.24]
hl2 = [-29.05, 37.40]
hl3 = [-30.85, 37.51]
hl4 = [-32.69, 37.62]
hl5 = [-34.64, 37.75]
hl6 = [-37.11, 37.97]
hl7 = [-39.41, 38.15]
hl8 = [-41.68, 38.33]
hl9 = [-44.05, 38.51]
hl10 = [-47.07, 38.79]
hl11 = [-49.85, 39.03]
hl12 = [-52.79, 39.27]
hl13 = [-56.07, 39.55]
hl14 = [-60.65, 40.00]
hl15 = [-65.46, 40.43]
hl16 = [-72.26, 41.04]
hl17 = [-82.80, 41.64]
hl18 = [-82.98, 40.59]
hl_PHcoo = [hl1, hl2, hl3, hl4, hl5, hl6, hl7, hl8, hl9, hl10, hl11, hl12, hl13, hl14, hl15, hl16, hl17, hl18]

################################### Mare Nubium + others (Mare in the fits) ###########################################
mare1 = [345.24, -26.60]
mare2 = [343.37, -26.42]
mare3 = [341.70, -26.29]
mare4 = [340.02, -26.15]
mare5 = [338.22, -25.99]
mare6 = [336.18, -25.76]
mare7 = [334.61, -25.65]
mare8 = [332.88, -25.50]
mare9 = [330.95, -25.30]
mare10 = [328.80, -25.04]
mare11 = [326.92, -24.86]
mare12 = [325.04, -24.68]
mare13 = [323.19, -24.51]
mare14 = [320.80, -24.21]
mare15 = [318.56, -23.95]
mare16 = [316.28, -23.69]
mare_PHcoo = [mare1, mare2, mare3, mare4, mare5, mare6, mare7, mare8, mare9, mare10, mare11, mare12, mare13, mare14,
              mare15, mare16]

# Inclination and emission angle of each region
################################################ Mare Imbrium (HL in the fits) ########################################
i_e_hl1 = [75, 51]
i_e_hl2 = [73, 52]
i_e_hl3 = [72, 53]
i_e_hl4 = [71, 54]
i_e_hl5 = [69, 55]
i_e_hl6 = [67, 57]
i_e_hl7 = [66, 58]
i_e_hl8 = [64, 60]
i_e_hl9 = [63, 61]
i_e_hl10 = [61, 63]
i_e_hl11 = [59, 65]
i_e_hl12 = [57, 68]
i_e_hl13 = [55, 70]
i_e_hl14 = [53, 73]
i_e_hl15 = [51, 77]
i_e_hl16 = [48, 82]
i_e_hl17 = [45, 90]
i_e_hl18 = [44, 90]
i_e_hl = [i_e_hl1, i_e_hl2, i_e_hl3, i_e_hl4, i_e_hl5, i_e_hl6, i_e_hl7, i_e_hl8, i_e_hl9, i_e_hl10, i_e_hl11, i_e_hl12,
          i_e_hl13, i_e_hl14, i_e_hl15, i_e_hl16, i_e_hl17, i_e_hl18]

################################### Mare Nubium + others (Mare in the fits) ###########################################
i_e_mare1 = [83, 27]
i_e_mare2 = [81, 28]
i_e_mare3 = [80, 29]
i_e_mare4 = [78, 30]
i_e_mare5 = [77, 32]
i_e_mare6 = [75, 33]
i_e_mare7 = [73, 34]
i_e_mare8 = [72, 35]
i_e_mare9 = [70, 37]
i_e_mare10 = [68, 38]
i_e_mare11 = [67, 40]
i_e_mare12 = [65, 41]
i_e_mare13 = [63, 43]
i_e_mare14 = [61, 45]
i_e_mare15 = [59, 46]
i_e_mare16 = [57, 48]
i_e_mare = [i_e_mare1, i_e_mare2, i_e_mare3, i_e_mare4, i_e_mare5, i_e_mare6, i_e_mare7, i_e_mare8, i_e_mare9,
            i_e_mare10, i_e_mare11, i_e_mare12, i_e_mare13, i_e_mare14, i_e_mare15, i_e_mare16]

# Delta position with respect to the center of the Moon of each offset in each region (empirical)
################################################ Mare Imbrium (HL in the fits) ########################################
delta_pos_hl = [[-344.34, 600.63], [-365.65, 601.82], [-383.85, 602.33], [-401.96, 602.84], [-420.60, 603.48],
                [-443.01, 604.92], [-463.09, 605.89], [-482.21, 606.62], [-501.16, 607.34], [-523.49, 608.79],
                [-542.87, 609.65], [-561.78, 610.35], [-580.91, 611.12], [-603.46, 612.67], [-623.01, 613.56],
                [-642.46, 614.41], [-661.69, 615.23], [-689.62, 618.14],]

#################################### Mare Nubium+Mare Hummorum (HL in the fits) ######################
delta_pos_mare = [[-248.09, -329.93], [-273.20, -327.63], [-295.50, -325.98], [-317.78, -324.33], [-341.28, -322.37],
                  [-367.86, -319.59], [-387.89, -318.50], [-409.79, -316.88], [-433.82, -314.70], [-460.30, -311.87],
                  [-482.81, -310.09], [-505.02, -308.36], [-526.21, -306.90], [-553.38, -303.77], [-577.84, -301.37],
                  [-601.99, -299.08],]

########################################## DATA FRAME TO RECOGNIZE THEM ############################################
################################################ Mare Imbrium (HL in the fits) ########################################
dic_hl_pos = {'ra off': np.asarray(delta_pos_hl)[:, 0], 'dec off': np.asarray(delta_pos_hl)[:, 1],
              'ra sele': np.asarray(hl_PHcoo)[:, 0], 'dec sele': np.asarray(hl_PHcoo)[:, 1],
              'inc angle': np.asarray(i_e_hl)[:, 0], 'emi angle': np.asarray(i_e_hl)[:, 1]}

df_hl_pos = pd.DataFrame(data=dic_hl_pos)

#################################### Mare Nubium+Mare Hummorum (HL in the fits) ######################
dic_mare_pos = {'ra off': np.asarray(delta_pos_mare)[:, 0], 'dec off': np.asarray(delta_pos_mare)[:, 1],
                'ra sele': np.asarray(mare_PHcoo)[:, 0], 'dec sele': np.asarray(mare_PHcoo)[:, 1],
                'inc angle': np.asarray(i_e_mare)[:, 0], 'emi angle': np.asarray(i_e_mare)[:, 1]}

df_mare_pos = pd.DataFrame(data=dic_mare_pos)

