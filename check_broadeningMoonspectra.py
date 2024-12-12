from typing import List

from moon_reflectance import *


def loop_broadening(tw, tf, w, f, alb, res_down, select_rv_arm, best_solar_arm, x_lims_arr, tf_err, f_err, tol=0.1,
                    dR=200, slit_width=0.5, n_poly_alb=3, n_poly_tf=6, loop=True, corr_data=False, **kwargs):
    # Beginning of the iteration --> initial guess
    tf_smooth = broad_moon_spectra(tw, tf, res_down)
    f_smooth = broad_moon_spectra(w, f, res_down)
    if not loop:
        if corr_data:
            raise ValueError("Try tomorrow, we still working on this...")
        else:
            meanWvl = np.mean(tw.data)
            fwhm_kernel = 1. / float(res_down) * meanWvl
            width_kernel = fwhm_kernel / (2. * np.sqrt(2. * np.log(2.)))
            width_kernel_variance = width_kernel / np.sqrt(2)
            solar_flux_variance = f_err**2
            lunar_flux_variance = tf_err**2
            convolved_solarFlux_variance = pyasl.broadGaussFast(w, solar_flux_variance, width_kernel_variance,
                                                                edgeHandling='firstlast', maxsig=None)
            convolved_lunarFlux_variance = pyasl.broadGaussFast(tw, lunar_flux_variance, width_kernel_variance,
                                                                edgeHandling='firstlast', maxsig=None)
            convolved_solarFlux_err = np.sqrt(convolved_solarFlux_variance/(2*np.sqrt(2*np.pi)*width_kernel_variance))
            convolved_lunarFlux_err = np.sqrt(convolved_lunarFlux_variance/(2*np.sqrt(2*np.pi)*width_kernel_variance))
            interp_solarFlux_err = interp1d(w, convolved_solarFlux_err, kind='slinear', fill_value="extrapolate")
            interp_lunarFlux_err = interp1d(tw, convolved_lunarFlux_err, kind='slinear', fill_value="extrapolate")
            f_convolved_err = interp_solarFlux_err(w.data)
            tf_convolved_err = interp_lunarFlux_err(tw.data)
            f_convolved_err = np.ma.masked_array(f_convolved_err, mask=w.mask)
            tf_convolved_err = np.ma.masked_array(tf_convolved_err, mask=tw.mask)
        alb_smooth, alb_smooth_err = geometricAlbedo(tf_smooth, f_smooth, slit_width, tf_convolved_err, f_convolved_err,
                                                     **kwargs)
        tf_smooth_norm = fit_cont_v2(tf_smooth, tw, n_poly_tf, 1, plot_cont=False)
        return alb_smooth, tf_smooth_norm, alb_smooth_err
    ##################################### loop #####################################################
    # new albedo --> We are considering so far tf_err = tf_broad_err
    alb_smooth, alb_smooth_err = geometricAlbedo(tf_smooth, f_smooth, slit_width, tf_err, f_err, **kwargs)
    alb_norm = []
    alb_smooth_norm = []
    std_lims = []
    std_smooth_lims = []
    for x_lims in x_lims_arr:
        m_zoom_tw = np.logical_and(tw.data > x_lims[0], tw.data < x_lims[1])
        m_zoom_w = np.logical_and(w.data > x_lims[0], w.data < x_lims[1])
        # reduce to the interval itself
        tw_reduced = tw.data[m_zoom_tw]
        alb_reduced = alb.data[m_zoom_tw]
        alb_smooth_reduced = alb_smooth.data[m_zoom_tw]
        tf_reduced = tf.data[m_zoom_tw]
        tf_smooth_reduced = tf_smooth.data[m_zoom_tw]
        # solar spectra
        w_reduced = w.data[m_zoom_w]
        f_reduced = f.data[m_zoom_w]
        f_smooth_reduced = f_smooth.data[m_zoom_w]
        # define the telluric mask again
        m_tw_now = mask_tellSpikes(tw_reduced, mask_all)
        m_w_now = mask_tellSpikes(w_reduced, mask_all)
        # define the correct intervals
        tw_to_analyze = np.ma.masked_array(tw_reduced, mask=m_tw_now)
        alb_to_analyze = np.ma.masked_array(alb_reduced, mask=m_tw_now)
        alb_smooth_to_analyze = np.ma.masked_array(alb_smooth_reduced, mask=m_tw_now)
        tf_to_analyze = np.ma.masked_array(tf_reduced, mask=m_tw_now)
        tf_smooth_to_analyze = np.ma.masked_array(tf_smooth_reduced, mask=m_tw_now)
        w_to_analyze = np.ma.masked_array(w_reduced, mask=m_w_now)
        f_to_analyze = np.ma.masked_array(f_reduced, mask=m_w_now)
        f_smooth_to_analyze = np.ma.masked_array(f_smooth_reduced, mask=m_w_now)
        # check if the whole interval is masked
        if not np.all(tw_to_analyze.mask):
            # print(f"The x_lims are {x_lims}")
            alb_norm_to_analyze = fit_cont_v2(alb_to_analyze, tw_to_analyze, n_poly_alb, 1, plot_cont=False)
            # if np.any(tw_to_analyze.mask):
            #     fig, ax = plt.subplots(2, 1)
            #     ax[0].plot(tw_to_analyze, alb_smooth_to_analyze, )
            #     ax[0].plot(tw_to_analyze, alb_to_analyze, c="b")
            #     ax[0].plot(tw_to_analyze.data[tw_to_analyze.mask], alb_smooth_to_analyze.data[tw_to_analyze.mask],
            #                c="grey")
            #     ax[1].plot(tw_to_analyze, tf_to_analyze, c="b")
            #     ax1 = ax[1].twinx()
            #     ax1.plot(w_to_analyze, f_to_analyze)
            #     fig.show()
            alb_smooth_norm_to_analyze = fit_cont_v2(alb_smooth_to_analyze, tw_to_analyze, n_poly_alb, 1,
                                                     plot_cont=False)
            std_interval = np.ma.std(alb_norm_to_analyze)
            std_smooth_interval = np.ma.std(alb_smooth_norm_to_analyze)
            alb_norm.append(alb_norm_to_analyze)
            alb_smooth_norm.append(alb_smooth_norm_to_analyze)
            std_lims.append(std_interval)
            std_smooth_lims.append(std_smooth_interval)
        else:  # the whole interval is masked
            alb_tell_to_analyze = fit_cont_v2(np.ma.masked_array(alb_to_analyze.data, mask=~m_tw_now),
                                              np.ma.masked_array(tw_to_analyze.data, mask=~m_tw_now), n_poly_alb, 1,
                                              plot_cont=False)
            alb_smooth_tell_to_analyze = fit_cont_v2(np.ma.masked_array(alb_smooth_to_analyze.data, mask=~m_tw_now),
                                                     np.ma.masked_array(tw_to_analyze.data, mask=~m_tw_now), n_poly_alb,
                                                     1, plot_cont=False)
            alb_norm.append(alb_tell_to_analyze)
            alb_smooth_norm.append(alb_smooth_tell_to_analyze)
            std_lims.append(np.nan)
            std_smooth_lims.append(np.nan)

    mask_both = np.logical_and(~np.isnan(np.asarray(std_lims)), ~np.isnan(np.asarray(std_smooth_lims)))
    std_lims_nonempty = np.asarray(std_lims)[mask_both]
    std_smooth_lims_nonempty = np.asarray(std_smooth_lims)[mask_both]
    std_lims = np.asarray(std_lims)
    std_smooth_lims = np.asarray(std_smooth_lims)
    print(np.median(std_smooth_lims_nonempty)/np.median(std_lims_nonempty))
    print(res_down)
    if np.median(std_smooth_lims_nonempty)/np.median(std_lims_nonempty) <= tol:
        tf_smooth_norm = fit_cont_v2(tf_smooth, tw, n_poly_tf, 1, plot_cont=False)
        return alb_norm, alb_smooth_norm, alb_smooth, tf_smooth_norm, std_lims, std_smooth_lims, x_lims_arr, res_down
    else:
        return loop_broadening(tw, tf, w, f, alb, res_down - dR, select_rv_arm, best_solar_arm, x_lims_arr, tf_err,
                               f_err, tol=tol, dR=dR, slit_width=slit_width, n_poly_alb=n_poly_alb)


def plot_spectra_partition(loc, xshoo_arm, select_rv_arm, best_solar_arm, x_lims_arr, res_down, idx_from=0, figure=None,
                           n_poly=5, tol=0.1, n_poly_alb=3, dR=200, slit_width=.5, idx_to=5, loop=True,
                           correct_edge=True):
    dlamba = 5 if xshoo_arm == "VIS" else 10
    notUVB = xshoo_arm == "VIS" or xshoo_arm == "NIR"
    ################################### RAW DATA ###################################
    if not notUVB:
        tf_arr, err_arr, airm_arr, date_arr, quants_col_arr, tw_arr = moon(loc, xshoo_arm, dim='1D',
                                                                           mask_tell_nd=mask_all, mode="pre_molecfit")
        tf_norm_arr, err_norm_arr, airm_norm_arr, date_norm_arr, quants_col_norm_arr, \
            tw_norm_arr = moon(loc, xshoo_arm, dim='1D', mask_tell_nd=mask_all, mode="pre_molecfit",
                               norm=True, n_poly=n_poly)
        tf, err, airm, date, quants, tw = tf_arr[idx_to], err_arr[idx_to], \
            airm_arr[idx_to], date_arr[idx_to], quants_col_arr[idx_to], tw_arr[idx_to]

    #################################### After molecfit ############################
    if notUVB:
        tf_arr, err_arr, airm_arr, date_arr, quants_col_arr, tw_arr = moon(loc, xshoo_arm, dim='1D',
                                                                           mask_tell_nd=mask_all)
        tf, err, airm, date, quants, tw = tf_arr[idx_to], err_arr[idx_to], \
            airm_arr[idx_to], date_arr[idx_to], quants_col_arr[idx_to], tw_arr[idx_to]

    ##################################### Correct edges ################################
    if xshoo_arm == 'VIS' and correct_edge:
        tf_uvb_arr, err_uvb_arr, airm_uvb_arr, \
            date_uvb_arr, quants_col_uvb_arr, tw_uvb_arr = moon(loc, 'UVB', dim='1D', mask_tell_nd=mask_all,
                                                                mode="pre_molecfit")
        tf_uvb, err_uvb, airm_uvb, date_uvb, quants_uvb, tw_uvb = tf_uvb_arr[idx_to], err_uvb_arr[idx_to], \
            airm_uvb_arr[idx_to], date_uvb_arr[idx_to], quants_col_uvb_arr[idx_to], tw_uvb_arr[idx_to]
        tf_nir_arr, err_nir_arr, airm_nir_arr, \
            date_nir_arr, quants_col_nir_arr, tw_nir_arr = moon(loc, 'NIR', dim='1D', mask_tell_nd=mask_all)
        tf_nir, err_nir, airm_nir, date_nir, quants_nir, tw_nir = tf_nir_arr[idx_to], err_nir_arr[idx_to], \
            airm_nir_arr[idx_to], date_nir_arr[idx_to], quants_col_nir_arr[idx_to], tw_nir_arr[idx_to]
    if correct_edge:
        edge_correct_tf = [[tw_uvb, tw, tw_nir], [tf_uvb, tf, tf_nir]] if xshoo_arm == 'VIS' else None
    else:
        edge_correct_tf = None
    ##################################### Apparent albedo ##############################
    ## Median resolution
    wv_alb, alb, f_interp, alb_err, f_interp_err = plot_global_reflectance([tw], [tf], [select_rv_arm], [err],
                                                                           solar_source='sun',
                                                                           best_solar_arm_arr=[best_solar_arm],
                                                                           more_offsets=True,
                                                                           plot_corrections=True if idx_to == 0 else False,
                                                                           best_method='median',
                                                                           x_lims=None, dlamba=dlamba,
                                                                           edge_correct_tf=edge_correct_tf,
                                                                           propagate_convErr=True,
                                                                           corr_data=False,
                                                                           )
    if idx_to == 0:
        fig_checkerr, ax_checkerr = plt.subplots(1, 1)
        ax_checkerr.plot(wv_alb, alb_err)
        fig_checkerr.show()
    ##################################### Apparent albedo broadened ##################################################
    if correct_edge:
        if xshoo_arm == 'VIS':
            lims = [[540, 590], [950, 1100]]
            tf = poly_edges(lims, edge_correct_tf[0], edge_correct_tf[1], [[2, 2], [3, 1]], p0=[5e-17, 4700],
                            method='pato_way', plot_correct=False, which_edge='vis_nir')[1]
        else:
            pass
    tf_norm = fit_cont_v2(tf, tw, n_poly, 1, plot_cont=False)

    ##################################### LOOPING OR NOT #############################################################
    if not loop:  # --> there is a chosen convolving resolution)
        a_br, tf_br_norm, a_br_err = loop_broadening(tw, tf, wv_alb, f_interp, alb, res_down, select_rv_arm,
                                                     best_solar_arm, x_lims_arr, err, f_interp_err,
                                                     slit_width=slit_width, n_poly_alb=n_poly_alb, n_poly_tf=n_poly,
                                                     loop=loop,)
        return a_br, tf_br_norm, a_br_err, date
    if isinstance(tol, list):
        a_norm_per_interval_arr = []
        # a_broad_norm_arr = []
        a_broad_norm_per_interval_arr = []
        a_broad_arr = []
        tf_broad_norm_arr = []
        std_list_original_arr = []
        std_list_after_criteria = []
        x_intervals_arr = []
        for to_i in tol:
            a_norm, a_br_norm, a_br, tf_br_norm, std_list, \
                std_list_smooth, x_intervals, final_resDown = loop_broadening(tw, tf, wv_alb, f_interp, alb,
                                                                              res_down, select_rv_arm, best_solar_arm,
                                                                              x_lims_arr, err, f_interp_err,
                                                                              tol=to_i, dR=dR, slit_width=slit_width,
                                                                              n_poly_alb=n_poly_alb, n_poly_tf=n_poly,)
            a_norm_per_interval_arr.append(a_norm)
            a_broad_norm_per_interval_arr.append(a_br_norm)
            a_broad_arr.append(a_br)
            tf_broad_norm_arr.append(tf_br_norm)
            std_list_original_arr.append(std_list)
            std_list_after_criteria.append(std_list_smooth)
            x_intervals_arr.append(x_intervals)
            print(f"################################################################################################")
    else:
        a_norm, a_br_norm, a_br, tf_br_norm, std_list, \
            std_list_smooth, x_intervals, final_resDown = loop_broadening(tw, tf, wv_alb, f_interp, alb, res_down,
                                                                          select_rv_arm, best_solar_arm, x_lims_arr,
                                                                          err, f_interp_err, tol=tol, dR=dR,
                                                                          slit_width=slit_width, n_poly_alb=n_poly_alb,
                                                                          n_poly_tf=n_poly,)
        tol = [tol]
        a_norm_per_interval_arr = [a_norm]
        # a_broad_norm_arr = [a_br_norm]
        a_broad_norm_per_interval_arr = [a_br_norm]
        a_broad_arr = [a_br]
        tf_broad_norm_arr = [tf_br_norm]
        std_list_original_arr = [std_list]
        std_list_after_criteria = [std_list_smooth]
        x_intervals_arr = [x_intervals]
        print(f"################################################################################################")
    a_norm_per_interval_arr = [a[idx_from:] for a in a_norm_per_interval_arr]
    a_broad_norm_per_interval_arr = [a_b[idx_from:] for a_b in a_broad_norm_per_interval_arr]
    if figure is not None:
        figure.suptitle(r'$\sigma_{median}^{0} =$' +
                        f"${np.ma.median(std_list_original_arr[0][~np.isnan(std_list_original_arr[0])]):.3f}$")
        # we partition the spectra
        for i_ax, (x_lims, axes) in enumerate(zip(x_lims_arr[idx_from:], figure.axes)):
            for i_tol, (alb_norm_criteria, alb_broad_norm_criteria, std_original, std_criteria,
                        to_i) in enumerate(zip(a_norm_per_interval_arr, a_broad_norm_per_interval_arr,
                                               std_list_original_arr, std_list_after_criteria, tol)):
                # original albedo
                # check the correct interval to plot
                m_zoom_alb_norm = np.logical_and(wv_alb.data > x_lims[0], wv_alb.data < x_lims[1])
                tw_reduced = tw.data[m_zoom_alb_norm]
                # add the telluric mask
                m_alb_norm_now = mask_tellSpikes(tw_reduced, mask_all)
                tw_alb_to_plot = np.ma.masked_array(tw_reduced, mask=m_alb_norm_now)
                if i_tol == 0:
                    if not np.all(tw_alb_to_plot.mask):
                        mask_nonempty1 = ~np.isnan(std_original)
                        frac_original = np.ma.std(alb_norm_criteria[i_ax]) / np.ma.median(std_original[mask_nonempty1])
                        axes.plot(tw_alb_to_plot, alb_norm_criteria[i_ax], color='k',
                                  label=r'$\sigma_{\lambda}^{0}/\sigma_{median}^{0}$' +
                                        f"$={frac_original:.3f}$")
                # broadened albedo
                # check the correct interval to plot
                m_zoom_alb_broad_norm = np.logical_and(wv_alb.data > x_lims[0], wv_alb.data < x_lims[1])
                tw_broad_reduced = tw.data[m_zoom_alb_broad_norm]
                # add the telluric mask
                m_alb_broad_norm_now = mask_tellSpikes(tw_broad_reduced, mask_all)
                tw_alb_broad_to_plot = np.ma.masked_array(tw_broad_reduced, mask=m_alb_broad_norm_now)
                if not np.all(tw_alb_to_plot.mask):
                    mask_nonempty_tol = np.logical_and(~np.isnan(std_original), ~np.isnan(std_criteria))
                    frac_broad = std_criteria[i_ax + idx_from] / np.ma.median(std_original[mask_nonempty_tol])
                    yes_OR_no = "REACHED" if frac_broad <= to_i else "not reached"
                    axes.plot(tw_alb_broad_to_plot, alb_broad_norm_criteria[i_ax],
                              label=r'$\epsilon = $' + f"${to_i}$: " +
                                    r'$\sigma_{\lambda}^{broad}/\sigma_{median}^{0}$ = ' +
                                    f"${frac_broad:.3f}$, {yes_OR_no}")
                # mask for the contaminated spectra
                # original albedo
                m_tellNorm_now = ~m_alb_norm_now
                wv_tellNorm_to_plot = np.ma.masked_array(tw_reduced.data, mask=m_tellNorm_now)
                # a part of the interval is masked
                if not np.all(wv_tellNorm_to_plot.mask):
                    if i_tol == 0:
                        axes.plot(wv_tellNorm_to_plot.data[~m_tellNorm_now],
                                  alb_norm_criteria[i_ax].data[~m_tellNorm_now], c='grey', alpha=0.5)
                # the whole wavelength interval is masked
                if np.all(~wv_tellNorm_to_plot.mask):
                    if i_tol == 0:
                        axes.plot(wv_tellNorm_to_plot.data[~m_tellNorm_now],
                                  alb_norm_criteria[i_ax].data[~m_tellNorm_now], c='grey', alpha=0.5)
            axes.legend(loc="upper left", fontsize=6)
            axes.grid()
            pass
        return wv_alb, alb, a_broad_arr, tf_norm, tf_broad_norm_arr, \
            std_list_original_arr, std_list_after_criteria, x_intervals_arr, final_resDown
    else:
        return wv_alb, alb, a_broad_arr, tf_norm, tf_broad_norm_arr, \
            std_list_original_arr, std_list_after_criteria, x_intervals_arr, final_resDown


def all_reflectance_perMare(tol_arr, i_arr, dR=50, correct_edge=True):
    resDown_listUVB = []
    resDown_listVIS = []
    resDown_listNIR = []
    wv_global = []
    alb_global = []
    tf_norm_global = []
    for i in i_arr:
        if i == 11:
            continue
        print(f"OFFSET: {i}")
        # UVB
        wvUVB, alUVB, al_brUVB_arr, tfUVB_norm, tf_brUVB_arr, stdUVB_original_arr, \
            stdUVB_after_criteria_arr, xUVB_arr, \
            final_resDownUVB = plot_spectra_partition("maria", "UVB", select_rv_UVB, best_solar_UVB, x_lims_uvb, 50_000,
                                                      idx_from=0, figure=None, tol=tol_arr, n_poly=7, dR=dR,
                                                      slit_width=.5, n_poly_alb=2, idx_to=i, correct_edge=correct_edge)

        # VIS
        wvVIS, alVIS, al_brVIS_arr, tfVIS_norm, tf_brVIS_arr, stdVIS_original_arr, \
            stdVIS_after_criteria_arr, xVIS_arr, \
            final_resDownVIS = plot_spectra_partition("maria", "VIS", select_rv_VIS, best_solar_VIS, x_lims_vis, 50_000,
                                                      idx_from=0, figure=None, tol=tol_arr, n_poly=8, dR=dR,
                                                      slit_width=.4, n_poly_alb=2, idx_to=i, correct_edge=correct_edge)
        # NIR
        wvNIR, alNIR, al_brNIR_arr, tfNIR_norm, tf_brNIR_arr, stdNIR_original_arr, \
            stdNIR_after_criteria_arr, xNIR_arr, \
            final_resDownNIR = plot_spectra_partition("maria", "NIR", select_rv_NIR, best_solar_NIR, x_lims_nir, 50_000,
                                                      idx_from=0, figure=None, tol=tol_arr, n_poly=10, dR=dR,
                                                      slit_width=.4, n_poly_alb=2, idx_to=i, correct_edge=correct_edge)
        # We sort all the quantities
        wv_global_idx = [wvUVB.data, wvVIS.data, wvNIR.data]
        alb_global_idx = [alUVB.data, alVIS.data, alNIR.data]
        tf_norm_global_idx = [tfUVB_norm.data, tfVIS_norm.data, tfNIR_norm.data]
        mask_global_idx = [wvUVB.mask, wvVIS.mask, wvNIR.mask]
        wv_global_idx = np.ma.masked_array(list(chain.from_iterable(wv_global_idx)),
                                           mask=list(chain.from_iterable(mask_global_idx)))
        alb_global_idx = np.ma.masked_array(list(chain.from_iterable(alb_global_idx)),
                                            mask=list(chain.from_iterable(mask_global_idx)))
        tf_norm_global_idx = np.ma.masked_array(list(chain.from_iterable(tf_norm_global_idx)),
                                                mask=list(chain.from_iterable(mask_global_idx)))

        resDown_listUVB.append(final_resDownUVB)
        resDown_listVIS.append(final_resDownVIS)
        resDown_listNIR.append(final_resDownNIR)
        wv_global.append(wv_global_idx)
        alb_global.append(alb_global_idx)
        tf_norm_global.append(tf_norm_global_idx)
    # Choose the minumum kernel for all the offsets
    common_resDownUVB = np.min(resDown_listUVB)
    common_resDownVIS = np.min(resDown_listVIS)
    common_resDownNIR = np.min(resDown_listNIR)
    alb_broad_global = []
    tf_broad_global = []
    alb_broad_err_global = []
    date_global = []
    for i in i_arr:
        print(f"OFFSET (NOW WE ARE CONVOLVING INTO A COMMON KERNEL BETWEEN OFFSETS): {i}")
        a_brUVB, tf_br_normUVB, a_br_errUVB, dateUVB = plot_spectra_partition("maria", "UVB", select_rv_UVB,
                                                                              best_solar_UVB, x_lims_uvb,
                                                                              common_resDownUVB, n_poly=7,
                                                                              slit_width=.5, n_poly_alb=2, idx_to=i,
                                                                              loop=False, correct_edge=correct_edge)
        a_brVIS, tf_br_normVIS, a_br_errVIS, dateVIS = plot_spectra_partition("maria", "VIS", select_rv_VIS,
                                                                              best_solar_VIS, x_lims_vis,
                                                                              common_resDownVIS, n_poly=8,
                                                                              slit_width=.4, n_poly_alb=2, idx_to=i,
                                                                              loop=False, correct_edge=correct_edge)
        a_brNIR, tf_br_normNIR, a_br_errNIR, dateNIR = plot_spectra_partition("maria", "NIR", select_rv_NIR,
                                                                              best_solar_NIR, x_lims_nir,
                                                                              common_resDownNIR, n_poly=10,
                                                                              slit_width=.4, n_poly_alb=2, idx_to=i,
                                                                              loop=False, correct_edge=correct_edge)
        alb_broad_global_idx = np.ma.masked_array(list(chain.from_iterable([a_brUVB.data, a_brVIS.data, a_brNIR.data])),
                                                  mask=list(chain.from_iterable([a_brUVB.mask, a_brVIS.mask,
                                                                                 a_brNIR.mask])))
        tf_broad_global_idx = np.ma.masked_array(list(chain.from_iterable([tf_br_normUVB.data, tf_br_normVIS.data,
                                                                           tf_br_normNIR.data])),
                                                 mask=list(chain.from_iterable([tf_br_normUVB.mask, tf_br_normVIS.mask,
                                                                                tf_br_normNIR.mask])))
        alb_broad_err_global_idx = np.ma.masked_array(list(chain.from_iterable([a_br_errUVB.data, a_br_errVIS.data,
                                                                                a_br_errNIR.data])),
                                                      mask=list(chain.from_iterable([a_br_errUVB.mask, a_br_errVIS.mask,
                                                                                     a_br_errNIR.mask])))
        date_global_idx = [dateUVB, dateVIS, dateNIR]

        alb_broad_global.append(alb_broad_global_idx)
        tf_broad_global.append(tf_broad_global_idx)
        alb_broad_err_global.append(alb_broad_err_global_idx)
        date_global.append(date_global_idx)
    return wv_global, alb_global, alb_broad_global, tf_norm_global, \
        tf_broad_global, alb_broad_err_global, date_global


def all_reflectance_perHighlands(tol_arr, i_arr, dR=50, correct_edge=True):
    resDown_listUVB = []
    resDown_listVIS = []
    wv_global = []
    alb_global = []
    tf_norm_global = []
    for i in i_arr:
        print(f"OFFSET: {i}")
        # UVB
        wvUVB, alUVB, al_brUVB_arr, tfUVB_norm, tf_brUVB_arr, stdUVB_original_arr, \
            stdUVB_after_criteria_arr, xUVB_arr, \
            final_resDownUVB = plot_spectra_partition("highlands", "UVB", select_rv_UVB, best_solar_UVB, x_lims_uvb, 50_000,
                                                      idx_from=0, figure=None, tol=tol_arr, n_poly=7, dR=dR,
                                                      slit_width=.5, n_poly_alb=2, idx_to=i, correct_edge=correct_edge)

        # VIS
        wvVIS, alVIS, al_brVIS_arr, tfVIS_norm, tf_brVIS_arr, stdVIS_original_arr, \
            stdVIS_after_criteria_arr, xVIS_arr, \
            final_resDownVIS = plot_spectra_partition("highlands", "VIS", select_rv_VIS, best_solar_VIS, x_lims_vis, 50_000,
                                                      idx_from=0, figure=None, tol=tol_arr, n_poly=8, dR=dR,
                                                      slit_width=.4, n_poly_alb=2, idx_to=i, correct_edge=correct_edge)
        # We sort all the quantities
        wv_global_idx = [wvUVB.data, wvVIS.data]
        alb_global_idx = [alUVB.data, alVIS.data]
        tf_norm_global_idx = [tfUVB_norm.data, tfVIS_norm.data]
        mask_global_idx = [wvUVB.mask, wvVIS.mask]
        wv_global_idx = np.ma.masked_array(list(chain.from_iterable(wv_global_idx)),
                                           mask=list(chain.from_iterable(mask_global_idx)))
        alb_global_idx = np.ma.masked_array(list(chain.from_iterable(alb_global_idx)),
                                            mask=list(chain.from_iterable(mask_global_idx)))
        tf_norm_global_idx = np.ma.masked_array(list(chain.from_iterable(tf_norm_global_idx)),
                                                mask=list(chain.from_iterable(mask_global_idx)))

        resDown_listUVB.append(final_resDownUVB)
        resDown_listVIS.append(final_resDownVIS)
        wv_global.append(wv_global_idx)
        alb_global.append(alb_global_idx)
        tf_norm_global.append(tf_norm_global_idx)
    # Choose the minumum kernel for all the offsets
    common_resDownUVB = np.min(resDown_listUVB)
    common_resDownVIS = np.min(resDown_listVIS)
    alb_broad_global = []
    tf_broad_global = []
    alb_broad_err_global = []
    date_global = []
    for i in i_arr:
        print(f"OFFSET (NOW WE ARE CONVOLVING INTO A COMMON KERNEL BETWEEN OFFSETS): {i}")
        a_brUVB, tf_br_normUVB, a_br_errUVB, dateUVB = plot_spectra_partition("highlands", "UVB", select_rv_UVB,
                                                                              best_solar_UVB, x_lims_uvb,
                                                                              common_resDownUVB, n_poly=7,
                                                                              slit_width=.5, n_poly_alb=2, idx_to=i,
                                                                              loop=False, correct_edge=correct_edge)
        a_brVIS, tf_br_normVIS, a_br_errVIS, dateVIS = plot_spectra_partition("highlands", "VIS", select_rv_VIS,
                                                                              best_solar_VIS, x_lims_vis,
                                                                              common_resDownVIS, n_poly=8,
                                                                              slit_width=.4, n_poly_alb=2, idx_to=i,
                                                                              loop=False, correct_edge=correct_edge)
        alb_broad_global_idx = np.ma.masked_array(list(chain.from_iterable([a_brUVB.data, a_brVIS.data])),
                                                  mask=list(chain.from_iterable([a_brUVB.mask, a_brVIS.mask])))
        tf_broad_global_idx = np.ma.masked_array(list(chain.from_iterable([tf_br_normUVB.data, tf_br_normVIS.data])),
                                                 mask=list(chain.from_iterable([tf_br_normUVB.mask,
                                                                                tf_br_normVIS.mask])))
        alb_broad_err_global_idx = np.ma.masked_array(list(chain.from_iterable([a_br_errUVB.data, a_br_errVIS.data])),
                                                      mask=list(chain.from_iterable([a_br_errUVB.mask,
                                                                                     a_br_errVIS.mask])))
        date_global_idx = [dateUVB, dateVIS]

        alb_broad_global.append(alb_broad_global_idx)
        tf_broad_global.append(tf_broad_global_idx)
        alb_broad_err_global.append(alb_broad_err_global_idx)
        date_global.append(date_global_idx)
    return wv_global, alb_global, alb_broad_global, tf_norm_global, \
        tf_broad_global, alb_broad_err_global, date_global


# UVB
range_lims_uvb = np.arange(374, 551, 1)  # 320
intervals_uvb = [lim for lim in range_lims_uvb if (lim - range_lims_uvb[0]) % 15 == 0]
x_lims_uvb = [[first, last] for first, last in zip(intervals_uvb[:-1], intervals_uvb[1:])]

# VIS
range_lims_vis = np.arange(554, 1_021, 1)  # 550
intervals_vis = [lim for lim in range_lims_vis if (lim - range_lims_vis[0]) % 10 == 0]
x_lims_vis = [[first, last] for first, last in zip(intervals_vis[:-1], intervals_vis[1:])]

# NIR
range_lims_nir = np.arange(1_050, 2_481, 1)
intervals_nir = [lim for lim in range_lims_nir if (lim - range_lims_nir[0]) % 40 == 0]
x_lims_nir = [[first, last] for first, last in zip(intervals_nir[:-1], intervals_nir[1:])]

tole_arr = [1.]
############################## HIGHLANDS #########################################
idx_hl = [0, 1]
WVGLOBAL_HL, ALBGLOBAL_HL, ALBBROADGLOBAL_HL, TFNORMGLOBAL_HL, \
    TFBROADNORMGLOBAL_HL, ALBBROADGLOBALERR_HL, DATESGLOBAL_HL = all_reflectance_perHighlands(tole_arr, idx_hl, dR=200,
                                                                                              correct_edge=False)
############################## MARE ##############################################
idx_mare = [0, 1]
WVGLOBAL_MARE, ALBGLOBAL_MARE, ALBBROADGLOBAL_MARE, TFNORMGLOBAL_MARE, \
    TFBROADNORMGLOBAL_MARE, ALBBROADGLOBALERR_MARE, DATESGLOBAL_MARE = all_reflectance_perMare(tole_arr, idx_mare,
                                                                                               dR=200,
                                                                                               correct_edge=False)
