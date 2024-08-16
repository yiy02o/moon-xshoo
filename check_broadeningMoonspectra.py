from typing import List

from moon_reflectance import *


def loop_broadening(tw, tf, w, f, alb, res_down, select_rv_arm, best_solar_arm, x_lims_arr, tf_err, tol=0.1, dR=200,
                    slit_width=0.5, n_poly_alb=3, n_poly_tf=6, loop=True):
    # Beginning of the iteration --> initial guess
    tf_smooth = broad_moon_spectra(tw, tf, res_down)
    f_smooth = broad_moon_spectra(w, f, res_down)

    # new albedo --> We are considering so far tf_err = tf_broad_err
    alb_smooth, alb_smooth_err = geometricAlbedo(tf_smooth, f_smooth, slit_width, tf_err, d_S_B=0.9938*u.AU,
                                                 solar_source='sun', d_E_B=399_184.62*u.km, alpha=100)
    if not loop:
        tf_smooth_norm = fit_cont_v2(tf_smooth, tw, n_poly_tf, 1, plot_cont=False)
        return alb_smooth, tf_smooth_norm, alb_smooth_err
    ##################################### loop #####################################################
    alb_norm = []
    alb_smooth_norm = []
    std_lims = []
    std_smooth_lims = []
    for x_lims in x_lims_arr:
        m_zoom_tw = np.logical_and(tw.data > x_lims[0], tw.data < x_lims[1])
        # reduce to the interval itself
        tw_reduced = tw.data[m_zoom_tw]
        alb_reduced = alb.data[m_zoom_tw]
        alb_smooth_reduced = alb_smooth.data[m_zoom_tw]
        # define the telluric mask again
        m_tw_now = mask_tellSpikes(tw_reduced, mask_all)
        # define the correct intervals
        tw_to_analyze = np.ma.masked_array(tw_reduced, mask=m_tw_now)
        alb_to_analyze = np.ma.masked_array(alb_reduced, mask=m_tw_now)
        alb_smooth_to_analyze = np.ma.masked_array(alb_smooth_reduced, mask=m_tw_now)
        # check if the whole interval is masked
        if not np.all(tw_to_analyze.mask):
            # print(f"The x_lims are {x_lims}")
            alb_norm_to_analyze = fit_cont_v2(alb_to_analyze, tw_to_analyze, n_poly_alb, 1, plot_cont=False)
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
                               tol=tol, dR=dR, slit_width=slit_width, n_poly_alb=n_poly_alb)


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
    if xshoo_arm == 'VIS':
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
    wv_alb, alb, alb_norm_arr, f_interp, alb_err = plot_global_reflectance([tw], [tf], [select_rv_arm], [err],
                                                                           solar_source='sun',
                                                                           best_solar_arm_arr=[best_solar_arm],
                                                                           more_offsets=True, plot_corrections=False,
                                                                           iteration=0, best_method='median',
                                                                           x_lims=None, dlamba=dlamba,
                                                                           edge_correct_tf=edge_correct_tf)
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
        a_br, tf_br_norm, a_br_err = loop_broadening(tw, tf, wv_alb, f_interp[0], alb, res_down, select_rv_arm,
                                                     best_solar_arm, x_lims_arr, err, slit_width=slit_width,
                                                     n_poly_alb=n_poly_alb, n_poly_tf=n_poly, loop=loop)
        return a_br, tf_br_norm, a_br_err
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
                std_list_smooth, x_intervals, final_resDown = loop_broadening(tw, tf, wv_alb, f_interp[0], alb,
                                                                              res_down, select_rv_arm, best_solar_arm,
                                                                              x_lims_arr, err, tol=to_i, dR=dR,
                                                                              slit_width=slit_width,
                                                                              n_poly_alb=n_poly_alb, n_poly_tf=n_poly)
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
            std_list_smooth, x_intervals, final_resDown = loop_broadening(tw, tf, wv_alb, f_interp[0], alb, res_down,
                                                                          select_rv_arm, best_solar_arm, x_lims_arr,
                                                                          err, tol=tol, dR=dR, slit_width=slit_width,
                                                                          n_poly_alb=n_poly_alb, n_poly_tf=n_poly)
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
        alb_norm_max, alb_norm_min = np.ma.median(alb_norm_arr) + 6*np.ma.std(alb_norm_arr), \
            np.ma.median(alb_norm_arr) - 3*np.ma.std(alb_norm_arr)
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
            axes.set_ylim(alb_norm_min, alb_norm_max)
            axes.legend(loc="upper left", fontsize=6)
            axes.grid()
            pass
        return wv_alb, alb, a_broad_arr, tf_norm, tf_broad_norm_arr, \
            std_list_original_arr, std_list_after_criteria, x_intervals_arr, final_resDown
    else:
        return wv_alb, alb, a_broad_arr, tf_norm, tf_broad_norm_arr, \
            std_list_original_arr, std_list_after_criteria, x_intervals_arr, final_resDown


def all_reflectance_perReg(tol_arr, i_arr, dR=50, correct_edge=True):
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
        wv_global_idx = [wvUVB, wvVIS, wvNIR]
        alb_global_idx = [alUVB, alVIS, alNIR]
        tf_norm_global_idx = [tfUVB_norm, tfVIS_norm, tfNIR_norm]
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
    for i in i_arr:
        print(f"OFFSET (NOW WE ARE CONVOLVING INTO A COMMON KERNEL BETWEEN OFFSETS): {i}")
        a_brUVB, tf_br_normUVB, a_br_errUVB = plot_spectra_partition("maria", "UVB", select_rv_UVB, best_solar_UVB,
                                                                     x_lims_uvb, common_resDownUVB, n_poly=7,
                                                                     slit_width=.5, n_poly_alb=2, idx_to=i, loop=False,
                                                                     correct_edge=correct_edge)
        a_brVIS, tf_br_normVIS, a_br_errVIS = plot_spectra_partition("maria", "VIS", select_rv_VIS, best_solar_VIS,
                                                                     x_lims_vis, common_resDownVIS, n_poly=8,
                                                                     slit_width=.4, n_poly_alb=2, idx_to=i, loop=False,
                                                                     correct_edge=correct_edge)
        a_brNIR, tf_br_normNIR, a_br_errNIR = plot_spectra_partition("maria", "NIR", select_rv_NIR, best_solar_NIR,
                                                                     x_lims_nir, common_resDownNIR, n_poly=10,
                                                                     slit_width=.4, n_poly_alb=2, idx_to=i, loop=False,
                                                                     correct_edge=correct_edge)
        alb_broad_global_idx = np.ma.masked_array(list(chain.from_iterable([a_brUVB, a_brVIS, a_brNIR])),
                                                  mask=list(chain.from_iterable([a_brUVB.mask, a_brVIS.mask,
                                                                                 a_brNIR.mask])))
        tf_broad_global_idx = np.ma.masked_array(list(chain.from_iterable([tf_br_normUVB, tf_br_normVIS,
                                                                           tf_br_normNIR])),
                                                 mask=list(chain.from_iterable([tf_br_normUVB.mask, tf_br_normVIS.mask,
                                                                                tf_br_normNIR.mask])))
        alb_broad_err_global_idx = np.ma.masked_array(list(chain.from_iterable([a_br_errUVB, a_br_errVIS,
                                                                                a_br_errNIR])),
                                                      mask=list(chain.from_iterable([a_br_errUVB.mask, a_br_errVIS.mask,
                                                                                     a_br_errNIR.mask])))
        alb_broad_global.append(alb_broad_global_idx)
        tf_broad_global.append(tf_broad_global_idx)
        alb_broad_err_global.append(alb_broad_err_global_idx)
    return wv_global, alb_global, alb_broad_global, tf_norm_global, tf_broad_global, alb_broad_err_global


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

tole_arr = [.35]
WVGLOBAL, ALBGLOBAL, ALBBROADGLOBAL, TFNORMGLOBAL, \
    TFBROADNORMGLOBAL, ALBBROADGLOBALERR = all_reflectance_perReg(tole_arr, np.arange(2), dR=150, correct_edge=False)

# tole_arr = [.8, .6, .4]
# UVB partition
# fig_uvb = plt.figure(figsize=(10, 10), layout='constrained')
# gs_uvb = fig_uvb.add_gridspec(5, 1, hspace=0, wspace=0)
# (ax1_uvb), (ax2_uvb), (ax3_uvb), (ax4_uvb), (ax5_uvb) = gs_uvb.subplots(sharey="row")
# np.random.seed(0)
# wv_uvb, al_uvb, al_br_uvb_arr, tf_uvb_norm, tf_br_uvb_arr, \
#     std_uvb_original_arr, std_uvb_after_criteria_arr, x_intervals_uvb_arr = plot_spectra_partition("maria", "UVB",
#                                                                                                   select_rv_UVB,
#                                                                                                   best_solar_UVB,
#                                                                                                   x_lims_uvb,
#                                                                                                   50_000, idx_from=10,
#                                                                                                   figure=fig_uvb,
#                                                                                                   tol=tole_arr,
#                                                                                                   n_poly=7, dR=50,
#                                                                                                   slit_width=.5,
#                                                                                                   n_poly_alb=2)
# fig_uvb.show()

#  VIS partition
# fig_vis = plt.figure(figsize=(10, 10), layout='constrained')
# gs_vis = fig_vis.add_gridspec(5, 1, hspace=0, wspace=0)
# (ax1_vis), (ax2_vis), (ax3_vis), (ax4_vis), (ax5_vis) = gs_vis.subplots(sharey="row")
# wv_vis, al_vis, al_br_vis_arr, tf_vis_norm, tf_br_vis_arr, \
#     std_vis_original_arr, std_vis_after_criteria_arr, x_intervals_vis_arr = plot_spectra_partition("maria", "VIS",
#                                                                                                    select_rv_VIS,
#                                                                                                    best_solar_VIS,
#                                                                                                    x_lims_vis,
#                                                                                                    50_000, idx_from=20,
#                                                                                                    figure=fig_vis,
#                                                                                                    tol=tole_arr,
#                                                                                                    n_poly=8,
#                                                                                                    dR=50, slit_width=.4,
#                                                                                                    n_poly_alb=2)
# fig_vis.show()

# NIR partition
# fig_nir = plt.figure(figsize=(10, 10), layout='constrained')
# gs_nir = fig_nir.add_gridspec(5, 1, hspace=0, wspace=0)
# (ax1_nir), (ax2_nir), (ax3_nir), (ax4_nir), (ax5_nir) = gs_nir.subplots(sharey="row")
# wv_nir, al_nir, al_br_nir_arr, tf_nir_norm, tf_br_nir_arr, \
#     std_nir_original_arr, std_nir_after_criteria_arr, x_intervals_nir_arr = plot_spectra_partition("maria", "NIR",
#                                                                                                    select_rv_NIR,
#                                                                                                    best_solar_NIR,
#                                                                                                    x_lims_nir,
#                                                                                                    50_000, idx_from=20,
#                                                                                                    figure=fig_nir,
#                                                                                                    tol=tole_arr,
#                                                                                                    n_poly=10,
#                                                                                                    dR=50, slit_width=.4,
#                                                                                                    n_poly_alb=2)
# fig_nir.show()

# std_after_criteria_all = [std_uvb_after_criteria_arr, std_vis_after_criteria_arr, std_nir_after_criteria_arr]
# std_original_all = [std_uvb_original_arr, std_vis_original_arr, std_nir_original_arr]
# x_intervals_all = [x_intervals_uvb_arr, x_intervals_vis_arr, x_intervals_nir_arr]

# fig_std = plt.figure(layout='constrained', figsize=(9, 6))
# subfigs_std = fig_std.subfigures(2, 1, wspace=0.01)
#
# axs_std1 = subfigs_std[0].subplots(1, 3)
# axs_std2 = subfigs_std[1].subplots(1, 3)
#
# for std_after_criteria_tol_arm, std_original_tol_arm, \
#         x_interval_tol_arm, color in zip(std_after_criteria_all, std_original_all, x_intervals_all, ["b", "y", "r"]):
#     for ax_std1, ax_std2, std_after_criteria_tol, std_original_tol, x_interval_tol, to \
#             in zip(axs_std1, axs_std2, std_after_criteria_tol_arm, std_original_tol_arm, x_interval_tol_arm, tole_arr):
#         non_nan_orig = ~np.isnan(std_original_tol)
#         non_nan_broad = ~np.isnan(std_after_criteria_tol)
#         rel_frac = np.asarray(std_after_criteria_tol[non_nan_broad]) / np.ma.median(std_original_tol[non_nan_orig])
#         ####################### relative sigma vs lambda #############################
#         ax_std1.plot(np.median(np.asarray(x_interval_tol)[non_nan_broad], axis=1), rel_frac, '.', color=color)
#         #for interval, frac in zip(x_interval_tol, rel_frac):
#         #    if frac <= to:
#         #        ax_std1.axvspan(interval[0], interval[1], facecolor="g", alpha=.3)
#         #    else:
#         #        continue
#         ####################### histogram relative sigma #############################
#         ax_std2.hist(rel_frac, 11, histtype="step", stacked="True", fill=False, color=color)
# ylims_std1 = [np.min([ax_std.get_ylim() for ax_std in axs_std1]), np.max([ax_std.get_ylim() for ax_std in axs_std1])]
# ylims_std2 = [np.min([ax_std.get_ylim() for ax_std in axs_std2]), np.max([ax_std.get_ylim() for ax_std in axs_std2])]
# for i_std, (ax_std1, ax_std2, to) in enumerate(zip(axs_std1, axs_std2, tole_arr)):
#     ####################### relative sigma vs lambda #############################
#     ax_std1.set_ylim(ylims_std1[0], ylims_std1[1])
#     # ax_std1.set_xlim(1_020, 2_500)
#     ax_std1.axhline(to, ls="--", color="k")
#     ax_std1.grid()
#     ax_std1.set_title(f"$\\epsilon = {to}$")
#     ax_std1.set_xlabel(f"$\\lambda$ (nm)")
#     ####################### histogram relative sigma #############################
#     ax_std2.set_ylim(ylims_std2[0], ylims_std2[1])
#     ax_std2.axvline(to, ls="--", color="k")
#     ax_std2.grid()
#     ax_std2.set_xlabel(r'$\sigma_{\lambda}^{broad}/\sigma_{median}^{0}$')
#     if i_std == 0:
#         ax_std1.set_ylabel(r'$\sigma_{\lambda}^{broad}/\sigma_{median}^{0}$')
#         ax_std2.set_ylabel(r'$N°$')
# fig_std.show()


# wv_global = [wv_uvb, wv_vis, wv_nir]
# alb_global = [al_uvb, al_vis, al_nir]
# tf_norm_global = [tf_uvb_norm, tf_vis_norm, tf_nir_norm]
# mask_global = [wv_uvb.mask, wv_vis.mask, wv_nir.mask]
# wv_global = np.ma.masked_array(list(chain.from_iterable(wv_global)), mask=list(chain.from_iterable(mask_global)))
# alb_global = np.ma.masked_array(list(chain.from_iterable(alb_global)), mask=list(chain.from_iterable(mask_global)))
# tf_norm_global = np.ma.masked_array(list(chain.from_iterable(tf_norm_global)), mask=list(chain.from_iterable(mask_global)))
# alb_broad_global = [np.ma.masked_array(list(chain.from_iterable([alb_br_uvb, alb_br_vis, alb_br_nir])),
#                                        mask=list(chain.from_iterable([alb_br_uvb.mask, alb_br_vis.mask,
#                                                                       alb_br_nir.mask])))
#                     for alb_br_uvb, alb_br_vis, alb_br_nir in zip(al_br_uvb_arr, al_br_vis_arr, al_br_nir_arr)]
# tf_broad_global = [np.ma.masked_array(list(chain.from_iterable([tf_br_uvb, tf_br_vis, tf_br_nir])),
#                                       mask=list(chain.from_iterable([tf_br_uvb.mask, tf_br_vis.mask, tf_br_nir.mask])))
#                    for tf_br_uvb, tf_br_vis, tf_br_nir in zip(tf_br_uvb_arr, tf_br_vis_arr, tf_br_nir_arr)]
# alb_norm_global = fit_cont_v2(alb_global, wv_global, 3, 1)
# alb_broad_norm_global = [fit_cont_v2(alb_br, wv_global, 3, 1) for alb_br in alb_broad_global]
