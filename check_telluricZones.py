from moon_reflectance import *
from astroquery.nist import Nist


def plot_spectra_partition(figure, loc, xshoo_arm, select_rv_arm, best_solar_arm, x_lims_arr, n_poly=5, n_lines=10,
                           linename="H I", plot_fits=False, dlamba_fit=0.1, lamba_stwin=None, J_stwin=None):

    def find_idx_closest(arr, k):
        idx_closest = (np.fabs(arr - k)).argmin()
        return idx_closest, np.fabs(arr[idx_closest] - k)

    def gaussian_fit(params, lamba_value):
        a, mu, sigma = params
        # a = 1 / (sigma*np.sqrt(2*np.pi))
        output = 1 + a * np.exp(-(lamba_value - mu)**2 / (2*sigma**2))
        return output

    def minimize_function(lamba_value, a, mu, sigma):
        params = a, mu, sigma
        return gaussian_fit(params, lamba_value)

    dlamba = 5 if xshoo_arm == "VIS" else 10
    notUVB = xshoo_arm == "VIS" or xshoo_arm == "NIR"

    ################################### RAW DATA ###################################
    f_raw_arr, err_raw_arr, airm_raw_arr, date_raw_arr, quants_col_raw_arr, wv_raw_arr = moon(loc, xshoo_arm, dim='1D',
                                                                                              mask_tell_nd=mask_all,
                                                                                              mode="pre_molecfit")
    idx_raw = np.random.randint(0, len(f_raw_arr))
    f_raw, err_raw, airm_raw, date_raw, quants_raw, wv_raw = f_raw_arr[idx_raw], err_raw_arr[idx_raw], \
        airm_raw_arr[idx_raw], date_raw_arr[idx_raw], quants_col_raw_arr[idx_raw], wv_raw_arr[idx_raw]
    try:
        f_rawNorm = fit_cont_v2(f_raw, wv_raw, n_poly, 1, plot_cont=False)
        wv_rawNorm = np.ma.masked_array(wv_raw.data, mask=f_rawNorm.mask)
    except:
        f_rawNorm = fit_cont_v2(np.ma.masked_array(f_raw, mask=np.repeat(False, len(f_raw))),
                                np.ma.masked_array(wv_raw, mask=np.repeat(False, len(wv_raw))),
                                n_poly, 1, plot_cont=False)
        wv_rawNorm = np.ma.masked_array(wv_raw, mask=f_rawNorm.mask)
    figure.suptitle(f"$\\sigma = {np.ma.std(f_rawNorm):.3f}$, " + r'FWHM$=2\sqrt{2log(2)}\sigma$, ' +
                    r'$R\sim\lambda/d\lambda$')

    #################################### After molecfit ############################
    if notUVB:
        f_mol_arr, err_mol_arr, airm_mol_arr, date_mol_arr, quants_col_mol_arr, wv_mol_arr = moon(loc, xshoo_arm,
                                                                                                  dim='1D',
                                                                                                  mask_tell_nd=mask_all)
        f_mol, err_mol, airm_mol, date_mol, quants_mol, wv_mol = f_mol_arr[idx_raw], err_mol_arr[idx_raw], \
            airm_mol_arr[idx_raw], date_mol_arr[idx_raw], quants_col_mol_arr[idx_raw], wv_mol_arr[idx_raw]
        f_molNorm = fit_cont_v2(f_mol, wv_mol, n_poly, 1, plot_cont=False)
        wv_molNorm = np.ma.masked_array(wv_mol.data, mask=f_molNorm.mask)
        f_molNorm_max, f_molNorm_min = np.ma.max(f_molNorm), np.ma.min(f_molNorm)

    ##################################### Apparent albedo ##############################
    # Median resolution
    wv_alb, alb, alb_norm_arr, f_interp, alb_err = plot_global_reflectance([wv_mol if notUVB else wv_raw],
                                                                           [f_mol if notUVB else f_raw],
                                                                           [select_rv_arm],
                                                                           [err_mol if notUVB else err_raw],
                                                                           solar_source='sun',
                                                                           best_solar_arm_arr=[best_solar_arm],
                                                                           more_offsets=True, plot_corrections=False,
                                                                           iteration=0, best_method='median',
                                                                           x_lims=None, dlamba=dlamba)
    if lamba_stwin is not None and J_stwin is not None:
        wv_alb_stwin, alb_stwin, alb_norm_stwin_arr, \
            f_interp_stwin, alb_err_stwin = plot_global_reflectance([wv_mol if notUVB else wv_raw],
                                                                    [f_mol if notUVB else f_raw], [select_rv_arm],
                                                                    [err_mol if notUVB else err_raw],
                                                                    lamba1_arr=[lamba_solar_VIS],
                                                                    data1_arr=[Js_tell_VIS], solar_source='solar_twin',
                                                                    best_solar_arm_arr=[best_solar_arm],
                                                                    more_offsets=True, plot_corrections=False,
                                                                    iteration=0, best_method='median', x_lims=None,
                                                                    dlamba=2)

    ##################### Solar flux interpolated normalized ####################################
    f_interp_norm = fit_cont_v2(f_interp[0], wv_alb, 6, 1, plot_cont=False)
    if lamba_stwin is not None and J_stwin is not None:
        f_interp_stwin_norm = fit_cont_v2(f_interp_stwin[0], wv_alb_stwin, 6, 1, plot_cont=False)

    f_rawNorm_max, f_rawNorm_min = np.ma.median(f_rawNorm) + 6*np.ma.std(f_rawNorm), \
        np.ma.median(f_rawNorm) - 6*np.ma.std(f_rawNorm)
    albNorm_max, albNorm_min = np.ma.median(alb_norm_arr) + 6*np.ma.std(alb_norm_arr), \
        np.ma.median(alb_norm_arr) - 3*np.ma.std(alb_norm_arr)

    if plot_fits:
        fig_fits, axs_fits = plt.subplots(1, 5, figsize=(10, 2), layout='constrained', sharey=True)
        axs_fits[2].set_xlabel(r'$\lambda - \lambda_{rest}$ (nm)')
    # we partition the spectra
    for idx_lims, (x_lims, axes) in enumerate(zip(x_lims_arr, figure.axes)):
        # RAW data
        m_zoom_raw = np.logical_and(wv_raw.data > x_lims[0], wv_raw.data < x_lims[1])
        m_raw_now = np.logical_or(wv_raw.mask, ~m_zoom_raw)
        wv_raw_to_plot = np.ma.masked_array(wv_raw.data, mask=m_raw_now)
        f_raw_to_plot = np.ma.masked_array(f_raw.data, mask=m_raw_now)
        # Normalized spectra
        m_zoom_rawNorm = np.logical_and(wv_rawNorm.data > x_lims[0], wv_rawNorm.data < x_lims[1])
        m_rawNorm_now = np.logical_or(wv_rawNorm.mask, ~m_zoom_rawNorm)
        wv_rawNorm_to_plot = np.ma.masked_array(wv_rawNorm.data, mask=m_rawNorm_now)
        f_rawNorm_to_plot = np.ma.masked_array(f_rawNorm.data, mask=m_rawNorm_now)
        # Solar spectra --> Sun
        m_zoom_sunNorm = np.logical_and(wv_alb.data > x_lims[0], wv_alb.data < x_lims[1])
        m_sunNorm_now = np.logical_or(wv_alb.mask, ~m_zoom_sunNorm)
        wv_sunNorm_to_plot = np.ma.masked_array(wv_alb.data, mask=m_sunNorm_now)
        f_sunNorm_to_plot = np.ma.masked_array(f_interp_norm.data, mask=m_sunNorm_now)
        # Solar spectra --> Solar twin
        if lamba_stwin is not None and J_stwin is not None:
            m_zoom_stwinNorm = np.logical_and(wv_alb_stwin.data > x_lims[0], wv_alb_stwin.data < x_lims[1])
            m_stwinNorm_now = np.logical_or(wv_alb_stwin.mask, ~m_zoom_stwinNorm)
            wv_stwinNorm_to_plot = np.ma.masked_array(wv_alb_stwin.data, mask=m_stwinNorm_now)
            f_stwinNorm_to_plot = np.ma.masked_array(f_interp_stwin_norm.data, mask=m_stwinNorm_now)
        if not np.all(wv_rawNorm_to_plot.mask):
            axes.plot(wv_rawNorm_to_plot, f_rawNorm_to_plot, c="k", alpha=0.8,
                      label=r'$\sigma_{\lambda}$' +
                            f"$={np.ma.std(f_molNorm_to_plot) if notUVB else np.ma.std(f_rawNorm_to_plot):.3f}$" if
                      not notUVB else None)
            axes.plot(wv_sunNorm_to_plot, f_sunNorm_to_plot + .2, c='g')
            if lamba_stwin is not None and J_stwin is not None:
                axes.plot(wv_stwinNorm_to_plot, f_stwinNorm_to_plot, c='y')
        axes.set_ylim(f_rawNorm_min, f_rawNorm_max)
        # axes.set_ylim(0.2, f_rawNorm_max)
        # axes.yaxis.set_major_formatter(plt.NullFormatter())

        # after molecfit data
        if notUVB:
            m_zoom_mol = np.logical_and(wv_mol.data > x_lims[0], wv_mol.data < x_lims[1])
            m_mol_now = np.logical_or(wv_mol.mask, ~m_zoom_mol)
            wv_mol_to_plot = np.ma.masked_array(wv_mol.data, mask=m_mol_now)
            f_mol_to_plot = np.ma.masked_array(f_mol.data, mask=m_mol_now)
            # Normalized spectra
            m_zoom_molNorm = np.logical_and(wv_molNorm.data > x_lims[0], wv_molNorm.data < x_lims[1])
            m_molNorm_now = np.logical_or(wv_molNorm.mask, ~m_zoom_molNorm)
            wv_molNorm_to_plot = np.ma.masked_array(wv_molNorm.data, mask=m_molNorm_now)
            f_molNorm_to_plot = np.ma.masked_array(f_molNorm.data, mask=m_molNorm_now)
            # mask for the contaminated spectra
            m_tellNorm_now = np.logical_or(~wv_molNorm.mask, ~m_zoom_molNorm)
            wv_tellNorm_to_plot = np.ma.masked_array(wv_molNorm.data, mask=m_tellNorm_now)
            f_tellNorm_to_plot = np.ma.masked_array(f_molNorm.data, mask=m_tellNorm_now)
            if not np.all(wv_tellNorm_to_plot.mask):
                axes.plot(wv_tellNorm_to_plot, f_tellNorm_to_plot, c='grey', alpha=0.5)
            if not np.all(wv_molNorm_to_plot.mask):
                axes.plot(wv_molNorm_to_plot, f_molNorm_to_plot - .2, label=r'$\sigma_{\lambda}$' +
                          f"$={np.ma.std(f_molNorm_to_plot) if notUVB else np.ma.std(f_rawNorm_to_plot):.3f}$")
        # albedo --> solar spectra
        m_zoom_albNorm = np.logical_and(wv_alb.data > x_lims[0], wv_alb.data < x_lims[1])
        m_albNorm_now = np.logical_or(wv_alb.mask, ~m_zoom_albNorm)
        wv_alb_to_plot = np.ma.masked_array(wv_alb.data, mask=m_albNorm_now)
        albNorm_to_plot = np.ma.masked_array(alb_norm_arr.data, mask=m_albNorm_now)
        # albedo --> twin solar spectra
        if lamba_stwin is not None and J_stwin is not None:
            m_zoom_albNorm_stwin = np.logical_and(wv_alb_stwin.data > x_lims[0], wv_alb_stwin.data < x_lims[1])
            m_albNorm_stwin_now = np.logical_or(wv_alb_stwin.mask, ~m_zoom_albNorm_stwin)
            wv_alb_to_plot_stwin = np.ma.masked_array(wv_alb_stwin.data, mask=m_albNorm_stwin_now)
            albNorm_stwin_to_plot = np.ma.masked_array(alb_norm_stwin_arr.data, mask=m_albNorm_stwin_now)
        axes1 = axes.twinx()
        if not np.all(wv_alb_to_plot.mask):
            #axes1.plot(wv_alb_to_plot, albNorm_to_plot, color='cyan')
            # axes.plot(wv_alb_to_plot, albNorm_to_plot, color='cyan')
            axes.plot(wv_alb_to_plot, albNorm_to_plot, color='r')
            if lamba_stwin is not None and J_stwin is not None:
                axes1.plot(wv_alb_to_plot_stwin, albNorm_stwin_to_plot - 0.04, color='m')
        axes1.set_ylim(albNorm_min, albNorm_max)
        axes1.yaxis.set_major_formatter(plt.NullFormatter())
        # shadowed areas
        #for rv_area in select_rv_arm:
        #    axes.fill_between(wv_rawNorm_to_plot, np.ma.median(alb_norm_arr) - 2*np.ma.std(alb_norm_arr), -2,
        #                      where=(wv_rawNorm_to_plot > rv_area[0]) & (wv_rawNorm_to_plot < rv_area[1]),
        #                      color="r", alpha=0.1)
        #for solar_line in best_solar_arm:
            #axes.fill_between(wv_rawNorm_to_plot, 2, np.ma.median(alb_norm_arr) - 2*np.ma.std(alb_norm_arr),
                              #where=(wv_rawNorm_to_plot > solar_line - dlamba) &
                                    #(wv_rawNorm_to_plot < solar_line + dlamba), color="g", alpha=0.1)
        # Nist
        Nist.clear_cache()
        try:
            table = Nist.query(x_lims[0]*u.nm, x_lims[1]*u.nm, linename=linename,
                               output_order="wavelength", wavelength_type="vac+air")
        except:
            axes.legend(loc="upper left", fontsize=6)
            axes.grid()
            continue
        wv_to_copy = wv_rawNorm_to_plot.copy() if xshoo_arm == "UVB" else wv_molNorm_to_plot.copy()
        f_to_copy = f_rawNorm_to_plot.copy() if xshoo_arm == "UVB" else f_molNorm_to_plot.copy()
        wv_to_copy = wv_to_copy.data[~wv_to_copy.mask]
        f_to_copy = f_to_copy.data[~f_to_copy.mask]
        try:
            wv_line_arr = []
            for i in np.arange(n_lines):
                mask_min = np.where(np.fabs(wv_to_copy[np.argmin(f_to_copy)] - wv_to_copy) <= dlamba_fit)
                wv_min = wv_to_copy[mask_min]
                f_min = f_to_copy[mask_min]
                # axes.plot(wv_min[np.argmin(f_min)], np.min(f_min), ".", c="r")
                if n_lines == 1:
                    p0 = -0.4, wv_min[np.argmin(f_min)], 0.2
                    p_opt, cov = curve_fit(minimize_function, wv_min, f_min, p0=p0, bounds=((-1, 0, 0),
                                                                                            (0, np.inf, np.inf)))
                    if plot_fits:
                        axs_fits[idx_lims].plot(wv_min - p_opt[1], f_min, 'b.-')
                        axs_fits[idx_lims].plot(wv_min - p_opt[1], gaussian_fit(p_opt, wv_min), color="r", ls="--",
                                                label=f"FWHM = {2*np.sqrt(2*np.log(2))*p_opt[-1]:.3f}, " + 
                                                      f"$R\\sim{p_opt[1]/(2*np.sqrt(2*np.log(2))*p_opt[-1]):.0f}$")
                        axs_fits[idx_lims].grid()
                        try:
                            axs_fits[idx_lims].set_ylim(np.ma.min(f_molNorm), 1.05)
                        except:
                            axs_fits[idx_lims].set_ylim(np.ma.min(f_rawNorm), 1.05)
                        axs_fits[idx_lims].set_xlim(-dlamba_fit, dlamba_fit)
                    print(p_opt)
                    axes.plot(wv_min, gaussian_fit(p_opt, wv_min), color="r",
                              label=f"FWHM = {2*np.sqrt(2*np.log(2))*p_opt[-1]:.3f}, " +
                                    f"$R\\sim{p_opt[1]/(2*np.sqrt(2*np.log(2))*p_opt[-1]):.0f}$", ls="--")
                wv_line_arr.append(wv_min[np.argmin(f_min)])
                wv_to_copy = np.delete(wv_to_copy, mask_min)
                f_to_copy = np.delete(f_to_copy, mask_min)
        except:
            wv_line_arr = []
        try:
            list_to_for = table["Observed"].data[~table["Observed"].mask].data
            list_species = table["Spectrum"].data[~table["Observed"].mask] if len(linename) != 1 else linename
            list_fik = table["fik"].data[~table["Observed"].mask]
            list_rel = table["Rel."].data[~table["Observed"].mask]
        except:
            list_to_for = table["Observed"]
            list_species = table["Spectrum"] if len(linename) != 1 else linename
            list_fik = table["fik"]
            list_rel = table["Rel."]
        not_repeat = []
        for wv_line in wv_line_arr:
            closest, dist = find_idx_closest(list_to_for, wv_line)
            line = list_to_for[closest]
            specie = list_species[closest] if len(linename) != 1 else linename[0]
            fik = list_fik[closest]
            rel = list_rel[closest]
            delta_y = 0.75 if "Fe" in specie else 1.1 if "Ti" in specie else 0.8 if "Ca" in specie else 0.9 if \
                "M" in specie else 1.15
            if dist > 0.02:
                continue
            if line not in not_repeat:
                axes.axvline(line, ymin=-2, ymax=2, color="k", ls="--", lw=1)
                axes.annotate(specie, (line + 0.03, delta_y))
                not_repeat.append(line)
            if plot_fits and n_lines == 1:
                axs_fits[idx_lims].annotate(specie, (wv_line - p_opt[1] + .03, .7))
                axs_fits[idx_lims].annotate(r'$\lambda_{rest} = $' + f"${wv_line:.2f}$",
                                            (wv_line - p_opt[1] + .01, .6), fontsize=7)
                axs_fits[idx_lims].annotate(r'$I_{rel} = $' + f"${rel}$", (wv_line - p_opt[1] - .08, .6), fontsize=7)
                axs_fits[idx_lims].legend(fontsize="xx-small")
        axes.legend(loc="upper left", fontsize=6)
        axes.grid()
    if plot_fits:
        fig_fits.show()
    pass


# UVB
range_lims_uvb = np.arange(320, 551, 1)
intervals_uvb = [lim for lim in range_lims_uvb if (lim - range_lims_uvb[0]) % 15 == 0]
x_lims_uvb = [[first, last] for first, last in zip(intervals_uvb[:-1], intervals_uvb[1:])]

# VIS
range_lims_vis = np.arange(553, 1_021, 1)
intervals_vis = [lim for lim in range_lims_vis if (lim - range_lims_vis[0]) % 10 == 0]
x_lims_vis = [[first, last] for first, last in zip(intervals_vis[:-1], intervals_vis[1:])]

# NIR
range_lims_nir = np.arange(1_050, 2_481, 1)
intervals_nir = [lim for lim in range_lims_nir if (lim - range_lims_nir[0]) % 40 == 0]
x_lims_nir = [[first, last] for first, last in zip(intervals_nir[:-1], intervals_nir[1:])]

# UVB partition
# fig_uvb = plt.figure(figsize=(10, 10), layout='constrained')
# gs_uvb = fig_uvb.add_gridspec(5, 1, hspace=0, wspace=0)
# (ax1), (ax2), (ax3), (ax4), (ax5) = gs_uvb.subplots(sharey="row")
# np.random.seed(0)
# plot_spectra_partition(fig_uvb, "maria", "UVB", select_rv_UVB, best_solar_UVB, x_lims_uvb[10:], n_poly=8,
#                        linename=["Fe", "Ca", "Na", "H I", "Mg", "Al", "Cr", "Sr", "Ti", "Si", "S", "Mn"])
# plt.show()

#  VIS partition
fig_vis = plt.figure(figsize=(10, 10), layout='constrained')
gs_vis = fig_vis.add_gridspec(5, 1, hspace=0, wspace=0)
(ax1), (ax2), (ax3), (ax4), (ax5) = gs_vis.subplots(sharey="row")
plot_spectra_partition(fig_vis, "maria", "VIS", select_rv_VIS, best_solar_VIS, x_lims_vis[30:], n_poly=8, n_lines=10,
                       linename=["Fe", "Ca", "Na", "Mg", "Al", "Cr", "Sr", "Ti", "Si", "S", "Mn"])
                       #linename=["Ca I", "Ca II"], plot_fits=True)
# ["Fe", "Ca", "Na", "H I", "Mg", "Al", "Cr", "Sr", "Ti", "Si", "S", "Mn"]
# ["Ca", "Na", "H I", "Mg", "Al", "Cr", "Sr"]
fig_vis.show()

# NIR partition
#fig_nir = plt.figure(figsize=(10, 10), layout='constrained')
#gs_nir = fig_nir.add_gridspec(5, 1, hspace=0, wspace=0)
#(ax1), (ax2), (ax3), (ax4), (ax5) = gs_nir.subplots(sharey="row")
#plot_spectra_partition(fig_nir, "maria", "NIR", select_rv_NIR, best_solar_NIR, x_lims_nir[35:], n_poly=10, n_lines=5,
#                       linename=["Fe", "Ca", "Na", "H I", "Mg", "Al", "Cr", "Sr", "Ti", "Si", "S", "Mn"],)
#plt.show()
