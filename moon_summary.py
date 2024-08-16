from moon_reflectance import *
from matplotlib import patheffects


def plot_all_results(axes, loc, xshoo_arm, select_rv_arm, best_solar_arm, x_lims, one_line=None):
    dlamba = 5 if xshoo_arm == "VIS" else 10

    ################################### RAW DATA ###################################
    f_raw_arr, err_raw_arr, airm_raw_arr, date_raw_arr, quants_col_raw_arr, wv_raw_arr = moon(loc, xshoo_arm, dim='1D',
                                                                                              mask_tell_nd=mask_all,
                                                                                              mode="pre_molecfit")
    idx_raw = np.random.randint(0, len(f_raw_arr))
    f_raw, err_raw, airm_raw, date_raw, quants_raw, wv_raw = f_raw_arr[idx_raw], err_raw_arr[idx_raw], \
        airm_raw_arr[idx_raw], date_raw_arr[idx_raw], quants_col_raw_arr[idx_raw], wv_raw_arr[idx_raw]
    m_zoom_raw = np.logical_and(wv_raw.data > x_lims[0], wv_raw.data < x_lims[1])
    m_raw_now = np.logical_or(wv_raw.mask, ~m_zoom_raw)
    wv_raw_to_plot = np.ma.masked_array(wv_raw.data, mask=m_raw_now)
    f_raw_to_plot = np.ma.masked_array(f_raw.data, mask=m_raw_now)
    axes[0].plot(wv_raw_to_plot, f_raw_to_plot)
    axes[0].annotate("Reduced data (before telluric correction)",
                     ((np.min(wv_raw_to_plot) + np.max(wv_raw_to_plot))/2,
                      (np.min(f_raw_to_plot) + np.max(f_raw_to_plot))/2 +
                      2*(np.max(f_raw_to_plot) - np.min(f_raw_to_plot))/5), size=35, ha="center",
                     path_effects=[patheffects.withStroke(linewidth=5, foreground="gray")])
    axes[0].xaxis.set_major_formatter(plt.NullFormatter())
    axes[0].yaxis.set_major_formatter(plt.NullFormatter())
    axes[0].grid()

    #################################### After molecfit ############################
    f_mol_arr, err_mol_arr, airm_mol_arr, date_mol_arr, quants_col_mol_arr, wv_mol_arr = moon(loc, xshoo_arm, dim='1D',
                                                                                              mask_tell_nd=mask_all)
    f_mol, err_mol, airm_mol, date_mol, quants_mol, wv_mol = f_mol_arr[idx_raw], err_mol_arr[idx_raw], \
        airm_mol_arr[idx_raw], date_mol_arr[idx_raw], quants_col_mol_arr[idx_raw], wv_mol_arr[idx_raw]
    m_zoom_mol = np.logical_and(wv_mol.data > x_lims[0], wv_mol.data < x_lims[1])
    m_mol_now = np.logical_or(wv_mol.mask, ~m_zoom_mol)
    wv_mol_to_plot = np.ma.masked_array(wv_mol.data, mask=m_mol_now)
    f_mol_to_plot = np.ma.masked_array(f_mol.data, mask=m_mol_now)
    axes[1].plot(wv_mol_to_plot, f_mol_to_plot)
    axes[1].annotate("Reduced data (after telluric correction)",
                     ((np.min(wv_mol_to_plot) + np.max(wv_mol_to_plot)) / 2,
                      (np.min(f_mol_to_plot) + np.max(f_mol_to_plot)) / 2 +
                      2*(np.max(f_mol_to_plot) - np.min(f_mol_to_plot)) / 5), size=35, ha="center",
                     path_effects=[patheffects.withStroke(linewidth=5, foreground="gray")])
    axes[1].xaxis.set_major_formatter(plt.NullFormatter())
    axes[1].yaxis.set_major_formatter(plt.NullFormatter())
    axes[1].grid()

    ##################################### Solar Model #################################
    path_hybridsolar005nm = '/home/yiyo/Downloads/solarhybrid005nm.csv'
    hybridsolar005nm = np.loadtxt(path_hybridsolar005nm, skiprows=1, delimiter=',')
    m_solar = np.logical_and(hybridsolar005nm.T[0] > x_lims[0], hybridsolar005nm.T[0] < x_lims[1])
    wv_solar = hybridsolar005nm.T[0][m_solar]
    f_solar = hybridsolar005nm.T[1][m_solar]
    axes[2].plot(wv_solar, f_solar)
    axes[2].annotate("Solar Hybrid model",
                     ((np.min(wv_solar) + np.max(wv_solar)) / 2,
                      (np.min(f_solar) + np.max(f_solar)) / 2 +
                      2 * (np.max(f_solar) - np.min(f_solar)) / 5), size=35, ha="center",
                     path_effects=[patheffects.withStroke(linewidth=5, foreground="gray")])
    axes[2].xaxis.set_major_formatter(plt.NullFormatter())
    axes[2].yaxis.set_major_formatter(plt.NullFormatter())
    axes[2].grid()

    ##################################### Apparent albedo ##############################
    # Median resolution
    wv_alb, alb, alb_norm_arr, f_interp = plot_global_reflectance([wv_mol], [f_mol], [select_rv_arm],
                                                                  solar_source='sun',
                                                                  best_solar_arm_arr=[best_solar_arm],
                                                                  more_offsets=True, plot_corrections=False,
                                                                  iteration=0, best_method='median',
                                                                  x_lims=x_lims, dlamba=dlamba)
    m_zoom_alb = np.logical_and(wv_alb.data > x_lims[0], wv_alb.data < x_lims[1])
    m_alb_now = np.logical_or(wv_alb.mask, ~m_zoom_alb)
    wv_interp_to_plot = np.ma.masked_array(wv_alb.data, mask=m_alb_now)
    f_interp_to_plot = np.ma.masked_array(f_interp[0].data, mask=m_alb_now)
    if one_line is not None:
        wv_alb_fix, alb_fix, alb_norm_fix_arr, f_interp_fix = plot_global_reflectance([wv_mol], [f_mol],
                                                                                      [select_rv_arm],
                                                                                      solar_source='sun',
                                                                                      best_solar_arm_arr=[[one_line]],
                                                                                      more_offsets=True,
                                                                                      plot_corrections=False,
                                                                                      iteration=0, best_method='median',
                                                                                      x_lims=x_lims, dlamba=dlamba)
        m_zoom_alb_fix = np.logical_and(wv_alb_fix.data > x_lims[0], wv_alb_fix.data < x_lims[1])
        m_alb_fix_now = np.logical_or(wv_alb_fix.mask, ~m_zoom_alb_fix)
        wv_interp_fix_to_plot = np.ma.masked_array(wv_alb_fix.data, mask=m_alb_fix_now)
        f_interp_fix_to_plot = np.ma.masked_array(f_interp_fix[0].data, mask=m_alb_fix_now)
    if one_line is None:
        axes[3].plot(wv_interp_to_plot, f_interp_to_plot)
    if one_line is not None:
        axes[3].plot(wv_interp_to_plot, f_interp_to_plot, label="Median best solars")
        axes[3].plot(wv_interp_fix_to_plot, f_interp_fix_to_plot, label=f"Fixed best solar: "
                                                                        f"[{one_line - dlamba}, {one_line + dlamba}]")
        axes[3].legend()
    axes[3].annotate("Broadened solar Hybrid model",
                     ((np.min(wv_alb) + np.max(wv_alb)) / 2,
                      (np.min(f_interp) + np.max(f_interp)) / 2 +
                      2 * (np.max(f_interp) - np.min(f_interp)) / 5), size=35, ha="center",
                     path_effects=[patheffects.withStroke(linewidth=5, foreground="gray")])

    if one_line is None:
        axes[4].plot(wv_alb, alb)
    if one_line is not None:
        axes[4].plot(wv_alb, alb, label="Median best solars")
        axes[4].plot(wv_alb_fix, alb_fix, label=f"Fixed best solar: [{one_line - dlamba}, {one_line + dlamba}]")
        axes[4].legend()
    axes[4].annotate("Apparent albedo (without photometric correction)",
                     ((np.min(wv_alb) + np.max(wv_alb)) / 2,
                      (np.min(alb) + np.max(alb)) / 2 +
                      2 * (np.max(alb) - np.min(alb)) / 5), size=35, ha="center",
                     path_effects=[patheffects.withStroke(linewidth=5, foreground="gray")])
    axes[3].xaxis.set_major_formatter(plt.NullFormatter())
    axes[3].yaxis.set_major_formatter(plt.NullFormatter())
    axes[3].grid()
    axes[4].set_xlabel(r"$\lambda$ (nm)", fontsize=25)
    axes[4].yaxis.set_major_formatter(plt.NullFormatter())
    axes[4].grid()


fig = plt.figure(layout='constrained', figsize=(40, 30))
subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[0.4, 0.6])
ax_vis = subfigs[0].subplots(5, 1)
axs_nir = subfigs[1].subplots(5, 1)
line_VIS = 710
line_NIR = 2_100
for a, l, s, b, x_lim, line in zip([ax_vis, axs_nir], ["VIS", "NIR"], [select_rv_VIS, select_rv_NIR],
                                   [best_solar_VIS, best_solar_NIR],
                                   [[line_VIS-5, line_VIS+5], [line_NIR-10, line_NIR+10]], [line_VIS, line_NIR]):
    plot_all_results(a, "maria", l, s, b, x_lim, one_line=line)
plt.show()

