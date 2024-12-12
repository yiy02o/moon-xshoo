from moon_reflectance import *


def plot_ratio(loc_BS, xshoo_arm, idx_BS=0, idx_DS=0, n_poly_BS=8, n_poly_DS=8):
    ################################### RAW DATA --> BS ###################################
    tf_BS_arr, err_BS_arr, airm_BS_arr, date_BS_arr, quants_col_BS_arr, tw_BS_arr = moon(loc_BS, xshoo_arm, dim='1D',
                                                                                         mask_tell_nd=mask_all,
                                                                                         mode="pre_molecfit")
    tf_BS, err_BS, airm_BS, date_BS, quants_BS, tw_BS = tf_BS_arr[idx_BS], err_BS_arr[idx_BS], \
        airm_BS_arr[idx_BS], date_BS_arr[idx_BS], quants_col_BS_arr[idx_BS], tw_BS_arr[idx_BS]

    ################################### RAW DATA --> DS ###################################
    tf_DS_arr, err_DS_arr, airm_DS_arr, date_DS_arr, quants_col_DS_arr, tw_DS_arr = moon("darkside", xshoo_arm,
                                                                                         dim='1D',
                                                                                         mask_tell_nd=mask_all,
                                                                                         mode="pre_molecfit")
    tf_DS, err_DS, airm_DS, date_DS, quants_DS, tw_DS = tf_DS_arr[idx_DS], err_DS_arr[idx_DS], \
        airm_DS_arr[idx_DS], date_DS_arr[idx_DS], quants_col_DS_arr[idx_DS], tw_DS_arr[idx_DS]

    # ratio
    tf_DS_norm = fit_cont_v2(tf_DS, tw_DS, n_poly_DS, 1, plot_cont=False)
    tf_BS_norm = fit_cont_v2(tf_BS, tw_BS, n_poly_BS, 1, plot_cont=False)
    ratio = tf_DS / tf_BS
    return tw_BS, tw_DS, tf_BS_norm, tf_DS_norm, ratio


def plot_brokenxaxis(nrow, ncol, selectX, X, Y, names=None, wv_rest=None):
    fig = plt.figure(figsize=(10, 4))
    width_ratios = list(np.ones(len(selectX)))
    for sps, select in zip(GridSpec(nrow, ncol, figure=fig), selectX):
        bax = brokenaxes(xlims=(selectX), subplot_spec=sps, width_ratios=width_ratios, despine=False, d=0.,
                         wspace=0.15)
        x_BS, x_DS = X
        y_BS, y_DS = Y   # y1 and y2 would be the object and solar twin data respectively
        bax.plot(x_BS, y_BS, c='b', label='Moonshine')
        bax.plot(x_DS, y_DS, c='k', label='Earthsine')
        bax.plot(x_DS, y_DS/y_BS, c='r', label='Ratio')
        bax.set_xlabel(r'$\lambda$ (nm)')
        bax.set_ylabel('Normalized intensity')
        if names is not None and wv_rest is not None:
            for _l, name in zip(wv_rest, names):
                # bax.plot([_l, _l], [0, 2], ls='--', c='grey')
                bax.axvline(_l, ls="--", c="grey")
                off_x = 0.04*2
                bax.annotate(name, xy=(_l + off_x, .4))
        bax.legend(loc='lower left', fontsize='x-small')
        bax.set_ylim(0.25, 1.2)
        bax.grid()
    fig.show()
    pass


# def earth_albedo(f_DS, f_BS, theta):
#     p_ratio = 1
#     f_a =


tw_BS_testVIS, tw_DS_testVIS, tf_BS_testVIS, tf_DS_testVIS, ratio_testVIS = plot_ratio("maria", "VIS")
tw_BS_testUVB, tw_DS_testUVB, tf_BS_testUVB, tf_DS_testUVB, ratio_testUVB = plot_ratio("maria", "UVB")
#tw_BS_test2, tw_DS_test2, tf_BS_test2, tf_DS_test2, ratio_test2 = plot_ratio("maria", "VIS", idx_DS=1)
#tw_BS_test3, tw_DS_test3, tf_BS_test3, tf_DS_test3, ratio_test3 = plot_ratio("maria", "VIS", idx_DS=2)
#tw_BS_test4, tw_DS_test4, tf_BS_test4, tf_DS_test4, ratio_test4 = plot_ratio("maria", "VIS", idx_DS=3)
lims = [853, 859]
m_BS_lims = np.ma.where(np.logical_and(tw_BS_testVIS.data > lims[0], tw_BS_testVIS.data < lims[1]))
m_DS_lims = np.ma.where(np.logical_and(tw_DS_testVIS.data > lims[0], tw_DS_testVIS.data < lims[1]))
#
# fig_ear, ax_ear = plt.subplots(1, 1, figsize=(10, 4))
# ax_ear.plot(tw_BS_test.data[m_BS_lims], tf_BS_test.data[m_BS_lims], label="Bright side")
# ax_ear.plot(tw_DS_test.data[m_DS_lims], tf_DS_test.data[m_DS_lims], label="Dark side")
# ax_ear.plot(tw_BS_test.data[m_BS_lims], tf_DS_test.data[m_DS_lims]/tf_BS_test.data[m_BS_lims], label="Ratio 1")
# #ax_ear.plot(tw_BS_test2.data[m_BS_lims], tf_DS_test2.data[m_DS_lims]/tf_BS_test2.data[m_BS_lims], label="Ratio 2")
# #ax_ear.plot(tw_BS_test3.data[m_BS_lims], tf_DS_test3.data[m_DS_lims]/tf_BS_test3.data[m_BS_lims], label="Ratio 3")
# #ax_ear.plot(tw_BS_test4.data[m_BS_lims], tf_DS_test4.data[m_DS_lims]/tf_BS_test4.data[m_BS_lims], label="Ratio 4")
# ax_ear.grid()
# ax_ear.legend()
# fig_ear.show()

################# UVB (part 1) ########################

lims_toLookUVB1 = [[374., 375.33], [374., 375.94], [374.82, 376.82], [403.81, 405.81]]
names_toLookUVB1 = ["Fe I", "Fe I", "Fe I", "Fe I"]
lamba_restUVB1 = [374.33, 374.94, 375.82, 404.81]
plot_brokenxaxis(1, 1, lims_toLookUVB1, [tw_BS_testUVB, tw_DS_testUVB],
                 [tf_BS_testUVB, tf_DS_testUVB], names=names_toLookUVB1, wv_rest=lamba_restUVB1)

################# UVB (part 2) ########################

lims_toLookUVB2 = [[421.67, 423.67], [442.56, 444.56], [444.47, 446.47], [444.58, 446.58]]
names_toLookUVB2 = ["Ca I", "Ca I", "Ca I", "Ca I"]
lamba_restUVB2 = [422.67, 443.56, 445.47, 445.58]
plot_brokenxaxis(1, 1, lims_toLookUVB2, [tw_BS_testUVB, tw_DS_testUVB],
                 [tf_BS_testUVB, tf_DS_testUVB], names=names_toLookUVB2, wv_rest=lamba_restUVB2)

################# UVB (part 3) ########################

lims_toLookUVB3 = [[491.05, 493.05], [495.60, 497.60], [508.07, 510.07], [512.36, 514.36], [542.25, 544.25]]
names_toLookUVB3 = ["Fe I", "Fe I", "Fe I", "Fe I", "Mn I"]
lamba_restUVB3 = [492.05, 496.60, 509.07, 513.36, 543.25]
plot_brokenxaxis(1, 1, lims_toLookUVB3, [tw_BS_testUVB, tw_DS_testUVB],
                 [tf_BS_testUVB, tf_DS_testUVB], names=names_toLookUVB3, wv_rest=lamba_restUVB3)

################# VIS (part 1) ########################
lims_toLookVIS = [[567.81, 569.81], [584.37, 586.37], [593.93, 595.93], [601.4, 603.4], [768.9, 770.9]]#, [853.2, 855.2]]
names_toLookVIS = ["Na I", "Ba II", "Fe I", "Fe I", "K I"]#, "Ca II"]
lamba_restVIS = [568.81, 585.37, 594.93, 602.4, 769.9]#, 854.2]

plot_brokenxaxis(1, 1, lims_toLookVIS[:], [tw_BS_testVIS, tw_DS_testVIS],
                 [tf_BS_testVIS, tf_DS_testVIS], names=names_toLookVIS[:], wv_rest=lamba_restVIS[:])
