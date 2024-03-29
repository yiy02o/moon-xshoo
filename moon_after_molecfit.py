import os
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib import patheffects
import numpy as np
from astropy.io import fits
import scipy
import pandas as pd
from astropy.stats import sigma_clip
from sklearn.cluster import KMeans


def compute_flux(path, dim):
    hdul = fits.open(path)
    date_obs = hdul[0].header['DATE-OBS']
    airm_initial = hdul[0].header['HIERARCH ESO TEL AIRM START']
    airm_final = hdul[0].header['HIERARCH ESO TEL AIRM END']
    airm = (airm_initial + airm_final) / 2
    flux = hdul[0].data.copy() if dim == '1D' else np.median(hdul[0].data.copy(), axis=0)
    err_flux = hdul[1].data.copy() if dim == '1D' else np.std(hdul[0].data.copy(), axis=0)
    hdul.close()
    return flux[1_000:], err_flux[1_000:], airm, date_obs   # to avoid saturation


def idp_subtract(path):
    hdul = fits.open(path)
    bin_table = hdul[1].data[0]
    obj_region = hdul[0].header['HIERARCH ESO OBS NAME']
    quants_col = hdul[1].columns
    return bin_table, obj_region, quants_col


def stdTell(obj_name, arm, dim='1D', counts=False, plot=False, ax=None, first=True):
    if arm == 'UVB':
        src = '/home/yiyo/moon_xshoo_pre_molecfit/'
        for dir in os.listdir(src):
            if obj_name in dir:
                src_complement = src + dir + '/reflex_end_products/'
                if len(os.listdir(src_complement)) > 1:
                    print('There are more than 1 std tell run over xshooter esoreflex {} arm'.format(arm))
                else:
                    run_time = 'first' if first else 'second'
                    new_src_complement = src_complement + os.listdir(src_complement)[0] + '/' + run_time + '/'
                    for arm_dir in os.listdir(new_src_complement):
                        if all(arm in word for word in os.listdir(new_src_complement + arm_dir + '/')):
                            for file in os.listdir(new_src_complement + arm_dir + '/'):
                                keyname = 'SLIT_MERGE' if counts else 'SLIT_FLUX_MERGE'
                                if keyname + dim in file:
                                    path_file = os.path.join(new_src_complement + arm_dir + '/', file)
                                    flux, err_flux, airm, date_obs = compute_flux(path_file, dim)
    else:
        src = '/home/yiyo/moon_xshoo_after_molecfit/'
        for dir in os.listdir(src):
            if obj_name in dir:
                src_complement = src + dir + '/reflex_end_products/molecfit/XSHOOTER/'
                if len(os.listdir(src_complement)) > 1:
                    print('There are more than 1 std tell run over xshooter esoreflex {} arm'.format(arm))
                else:
                    new_src_complement = src_complement + os.listdir(src_complement)[0] + '/'
                    for arm_dir in os.listdir(new_src_complement):
                        keyname = 'SLIT_MERGE' if counts else 'SLIT_FLUX_MERGE'
                        if keyname in arm_dir and arm in arm_dir:
                            filename = os.listdir(new_src_complement + arm_dir + '/')[0]
                            path_file = os.path.join(new_src_complement + arm_dir + '/', filename)
                            flux, err_flux, airm, date_obs = compute_flux(path_file, '1D')

    src = '/home/yiyo/moon_xshoo_pre_molecfit/'
    for dir in os.listdir(src):
        if obj_name in dir:
            src_complement = src + dir + '/reflex_end_products/'
            if len(os.listdir(src_complement)) > 1:
                print('There are more than 1 std tell run over xshooter esoreflex {} arm'.format(arm))
            else:
                run_time = 'first' if first else 'second'
                new_src_complement = src_complement + os.listdir(src_complement)[0] + '/' + run_time + '/'
                for arm_dir in os.listdir(new_src_complement):
                    if all(arm in word for word in os.listdir(new_src_complement + arm_dir + '/')):
                        for file in os.listdir(new_src_complement + arm_dir + '/'):
                            if 'IDP' in file:
                                # the info of our data
                                path_file = os.path.join(new_src_complement + arm_dir + '/', file)
                                hdul = fits.open(path_file)
                                bin_table = hdul[1].data[0]
                                obj_region = hdul[0].header['HIERARCH ESO OBS NAME']
                                quants_col = hdul[1].columns
                            else:
                                pass
    if plot:
        wave_length = bin_table['WAVE']
        if arm == 'UVB':
            c = 'b'
        if arm == 'VIS':
            c = 'yellow'
        if arm == 'NIR':
            c = 'r'
            if counts:
                y_label = quants_col['FLUX REDUCED'].unit
            else:
                y_label = 'erg ' + r'$cm^{-2} s^{-1} \AA^{-1}$'
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(obj_name + ' ' + date_obs, fontsize=16)
        ax.plot(wave_length[1000:], flux[1000:], c=c, label=arm, lw=0.5)
        ax.set_xscale('log')
    else:
        return flux, err_flux, airm, date_obs, bin_table, obj_region, quants_col


def norm_slope(y, x, n):
    z = np.ma.polyfit(x, y, n)
    p = np.poly1d(z)
    y_poly = p(x.data)
    y_norm = y / y_poly
    return y_poly


def mask_tellSpikes(lamba, X_n):
    mask_nd = np.asarray([np.logical_and(lamba > r[0], lamba < r[1]) for r in X_n])
    mask_reduced = np.logical_or.reduce(mask_nd)
    return mask_reduced


def moon(obj_name, arm, dim='1D', counts=False, plot=False, figure=None, idx_arr=None, wv_lims=None,
         arm_together=False, mask_tell_nd=None, norm=False, n_poly=4, mol_earth_atm=False):
    """The UVB data products (no molecfit correction) are saved in the directory
    /home/yiyo/moon_xshoo_pre_molecfit/. Conversely, VIS and NIR arm must be corrected with molecfit
    and are located in the directory /home/yiyo/moon_xshoo_after_molecfit. Most of the conditionals
    used in the code to search the files are a consequence of how the directories, subdirectories and
    files are sorted. So to find the data, an obj_name (--> 'maria' for example) and an
    arm (--> 'UVB') has to be provided. Also, you must specify if you want to work with the 1D or 2D
    merged spectrum (in the case of 2D spectrum, the median along the slit is computed for the equivalent
    one dimensional flux) and if you want to work with the counts or physical units (boolean).

    You also can plot the spectrum, and a figure must be provided. You can index between the number of
    observations (if you don't want to plot all and look for one in particular, [1, 4, 9] for example).
    If you wan t zoom the spectrum you can provide the range of wavelength that you want to see more in
    detail (in nm units). the arm_together parameter is just if you want to plot al arm parts of the spectra
    with a single colour (in this case is in black)."""

    if arm == 'UVB' or counts:
        src = '/home/yiyo/moon_xshoo_pre_molecfit/'
        for directory in os.listdir(src):
            if obj_name in directory:
                src_complement = src + directory + '/reflex_end_products/'
                list_obs_files = os.listdir(src_complement)
                if idx_arr is None:
                    idx_files = list_obs_files
                else:
                    idx_files = [list_obs_files[idx] for idx in idx_arr]
                flux_arr_per_obs = []
                err_flux_arr_per_obs = []
                airm_arr_per_obs = []
                date_arr_per_obs = []
                for subdir in idx_files:
                    # each subdir represents one single observation
                    subsubdir_list = os.listdir(src_complement + subdir + '/')
                    for subsubdir in subsubdir_list:
                        # searching in the arm subdirectories
                        if subsubdir == 'README':
                            continue
                        if all(arm in word for word in os.listdir(src_complement + subdir + '/' + subsubdir + '/')):
                            for file in os.listdir(src_complement + subdir + '/' + subsubdir + '/'):
                                keyname = 'SLIT_MERGE' if counts else 'SLIT_FLUX_MERGE'
                                if keyname + dim in file:
                                    path_file = os.path.join(src_complement + subdir + '/' + subsubdir + '/', file)
                                    flux, err_flux, airm, date_obs = compute_flux(path_file, dim)
                                    flux_arr_per_obs.append(flux)
                                    err_flux_arr_per_obs.append(err_flux)
                                    airm_arr_per_obs.append(airm)
                                    date_arr_per_obs.append(date_obs)
                                else:
                                    pass
                        else:
                            pass
            else:
                pass
    else:
        src = '/home/yiyo/moon_xshoo_after_molecfit/'
        for directory in os.listdir(src):
            if obj_name in directory:
                src_complement = src + directory + '/'
                for dim_directory in os.listdir(src_complement):
                    if dim in dim_directory:
                        new_src_complement = src_complement + dim + '/'
                        for arm_dir in os.listdir(new_src_complement):
                            if arm in arm_dir:
                                pre_file_src = new_src_complement + arm_dir + '/reflex_end_products/molecfit/XSHOOTER/'
                                list_obs_files = os.listdir(pre_file_src)
                                if idx_arr is None:
                                    idx_files = list_obs_files
                                else:
                                    idx_files = [list_obs_files[idx] for idx in idx_arr]
                                flux_arr_per_obs = []
                                err_flux_arr_per_obs = []
                                airm_arr_per_obs = []
                                date_arr_per_obs = []
                                earth_atm_abundance = []
                                for subdir in idx_files:
                                    dir_src = os.path.join(pre_file_src, subdir)
                                    if len(os.listdir(dir_src)) == 2:
                                        for subsubdir in os.listdir(dir_src):
                                            if subsubdir == 'README':
                                                continue
                                            else:
                                                subdir_src = os.path.join(dir_src, subsubdir)
                                                file_name = os.listdir(subdir_src + '/')[0]
                                                file_src = os.path.join(subdir_src + '/', file_name)
                                                flux, err_flux, airm, date_obs = compute_flux(file_src, '1D')
                                                flux_arr_per_obs.append(flux)
                                                err_flux_arr_per_obs.append(err_flux)
                                                airm_arr_per_obs.append(airm)
                                                date_arr_per_obs.append(date_obs)
                                                earth_atm_abundance.append(file_src)
                                    else:
                                        print('There are more than 1 subdirectory (1 + README file should be expected')
                                if mol_earth_atm:
                                    return earth_atm_abundance

                            else:
                                pass
                    else:
                        pass
            else:
                pass
    xshoo_pre_molec = '/home/yiyo/moon_xshoo_pre_molecfit/'
    for directory in os.listdir(xshoo_pre_molec):
        if obj_name in directory:
            print(directory)
            src_complement = xshoo_pre_molec + directory + '/reflex_end_products/'
            list_obs_files = os.listdir(src_complement)
            if idx_arr is None:
                idx_files = list_obs_files
            else:
                idx_files = [list_obs_files[idx] for idx in idx_arr]
            bin_table_arr_per_obs = []
            obj_region_arr_per_obs = []
            quants_col_arr_per_obs = []
            for subdir in idx_files :
                # each subdir represents one single observation
                for subsubdir in os.listdir(src_complement + subdir + '/'):
                    # searching in the arm subdirectories
                    if subsubdir == 'README':
                        continue
                    if all(arm in word for word in os.listdir(src_complement + subdir + '/' + subsubdir + '/')):
                        for file in os.listdir(src_complement + subdir + '/' + subsubdir + '/'):
                            keyname = 'IDP'
                            if keyname in file:
                                path_file = os.path.join(src_complement + subdir + '/' + subsubdir + '/', file)
                                bin_table, obj_region, quants_col = idp_subtract(path_file)
                                bin_table_arr_per_obs.append(bin_table)
                                obj_region_arr_per_obs.append(obj_region)
                                quants_col_arr_per_obs.append(quants_col)
                            else:
                                pass
                    else:
                        pass
        else:
            pass
    wave_length = [b['WAVE'][1_000:] for b in bin_table_arr_per_obs]  # In the same arm the wv range would be the same

    # if we want to normalize
    poly_to_norm = [norm_slope(f, wv, n_poly) for (f, wv) in zip(flux_arr_per_obs, wave_length)]
    if norm:
        flux_arr_per_obs = [f/p for (f, p) in zip(flux_arr_per_obs, poly_to_norm)]
        err_flux_arr_per_obs = [err/np.abs(p) for (err, p) in zip(err_flux_arr_per_obs, poly_to_norm)]

    # telluric mask
    if mask_tell_nd is not None:
        maskTell1D = [mask_tellSpikes(wv, mask_tell_nd) for wv in wave_length]
        flux_arr_per_obs = [np.ma.masked_array(f, m) for (f, m) in zip(flux_arr_per_obs, maskTell1D)]
        err_flux_arr_per_obs = [np.ma.masked_array(err, m) for (err, m) in zip(err_flux_arr_per_obs, maskTell1D)]
        wave_length = [np.ma.masked_array(wv, m) for (wv, m) in zip(wave_length, maskTell1D)]
    if wv_lims is not None:
        min_lim, max_lim = wv_lims
        mask_lims = [~np.logical_and(wv >= min_lim, wv <= max_lim) for wv in wave_length]
        flux_arr_per_obs = [np.ma.masked_array(f, m) for (f, m) in zip(flux_arr_per_obs, mask_lims)]
        err_flux_arr_per_obs = [np.ma.masked_array(err, m) for (err, m) in zip(err_flux_arr_per_obs, mask_lims)]
        wave_length = [np.ma.masked_array(wv, m) for (wv, m) in zip(wave_length, mask_lims)]

    # if we want to plot
    if plot:
        if arm == 'UVB':
            c = 'b'
        if arm == 'VIS':
            c = 'yellow'
        if arm == 'NIR':
            c = 'r'
        if arm_together:
            c = 'k'
        for f, wv, q, date, ax, p in zip(flux_arr_per_obs, wave_length, quants_col_arr_per_obs, date_arr_per_obs,
                                         figure.axes, poly_to_norm):
            if counts:
                y_label = q['FLUX_REDUCED'].unit
            else:
                pass
                y_label = 'erg ' + r'$cm^{-2} s^{-1} \AA^{-1}$'
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_xlabel(r'$\lambda$ (nm)', fontsize=10)
            ax.tick_params(labelsize=10)
            #if wv_lims is not None:
            #    min_lim, max_lim = wv_lims
            #    mask_zoom = np.logical_and(wv >= min_lim, wv <= max_lim)
            #    wv_zoom = wv[mask_zoom]
            #    f_zoom = f[mask_zoom]
            #    ax.plot(wv_zoom, f_zoom, c=c, label=date, lw=0.5)
            #else:
            ax.plot(wv, f, c=c, label=date, lw=0.5)
            ax.set_ylim(0)
            if not norm:
                ax.plot(wv, p, c='r', lw=1.5)
            ax.legend()
    else:
        return flux_arr_per_obs, err_flux_arr_per_obs, airm_arr_per_obs, date_arr_per_obs, quants_col_arr_per_obs, \
            wave_length


def lambda_ranges(obj_name, arm, n, s, plots_fit_cont, **kwargs):
    f, sigma, airm, date, q_col, lamba = moon(obj_name, arm, norm=True, n_poly=n, **kwargs)
    f_clip = [sigma_clip(flux, sigma=s, stdfunc=scipy.stats.iqr) for flux in f]
    m_clip = [f_c.mask for f_c in f_clip]
    # lamba = [np.ma.masked_array(l.data, mask=m) for (l, m) in zip(lamba, m_clip)]
    m_clip = np.all(m_clip, axis=0)
    lamba = [np.ma.masked_array(l.data, mask=m_clip) for l in lamba]

    if plots_fit_cont:
        plt.clf()
        fig, ax = plt.subplots(figsize=(25, 5))
        for f_c, wv in zip(f_clip, lamba):
            ax.plot(wv, f_c, label=date, lw=0.5)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Normalized flux', fontsize=14)
        ax.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
        plt.show()
    return lamba


def fit_the_cont(obj_name, arm, n, s, n_sigma, plots_fit_cont=True, median_region=False, **kwargs):
    if obj_name == 'maria':
        moon_loc = 'Nubium'
    if obj_name == 'highlands':
        moon_loc = 'Imbrium'
    if obj_name == 'darkside':
        moon_loc = 'Fecundidatis'
    f, sigma, airm, date_obs = moon(obj_name, arm, **kwargs)[0:4]
    lamba = lambda_ranges(obj_name, arm, n, s, plots_fit_cont, **kwargs)
    poly_fit = [norm_slope(flux, wv, n) for (flux, wv) in zip(f, lamba)]
    if plots_fit_cont:
        fig_cont, ax_cont = plt.subplots(1, 1, figsize=(15, 5), layout='constrained')
        offset = 0
        for poly, wv, flux in zip(poly_fit[:2], lamba[:2], f[:2]):
            ax_cont.plot(wv, poly + offset, lw=0.5, c='r')
            ax_cont.plot(wv, flux + offset, lw=0.5, c='k')
            offset = offset + 0.5*1e-13
        ax_cont.set_ylabel('Continuoum fit', fontsize=14)
        ax_cont.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
        ax_cont.set_title('Masked lines', fontsize=20)
        plt.show()
        plt.clf()

    f_norm = [flux/poly for (flux, poly) in zip(f, poly_fit)]
    sigma_norm = [err/poly for (err, poly) in zip(sigma, poly_fit)]
#
    #f_median = np.ma.masked_where(mask_median, np.ma.median(f_norm, axis=0))
    #lamba_median = np.ma.masked_where(mask_median, lamba[0].data)
    #sigma_median = np.ma.masked_where(mask_median, np.ma.std(f_norm, axis=0))
    #poly_median = np.ma.masked_where(mask_median, np.ma.median(poly_fit, axis=0))
    if plots_fit_cont:
        fig_norm, ax_norm = plt.subplots(figsize=(25, 5))
        for f_n, wv in zip(f_norm, lamba):
            ax_norm.plot(wv.data, f_n, lw=0.5)
            #ax_norm.set_xscale('log')
        ax_norm.set_ylabel('Normalized flux', fontsize=14)
        ax_norm.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
        ax_norm.set_title(f"{moon_loc} {arm} final flux normalization", fontsize=20)
        plt.show()

    mask_norm = [np.ma.getmask(f_n) for f_n in f_norm]
    mask_median = np.all(mask_norm, axis=0)
    f_median = np.ma.masked_where(mask_median, np.ma.median(f_norm, axis=0))
    lamba_median = np.ma.masked_where(mask_median, lamba[0].data)
    sigma_median = np.ma.masked_where(mask_median, np.ma.std(f_norm, axis=0))
    poly_median = np.ma.masked_where(mask_median, np.ma.median(poly_fit, axis=0))

    if plots_fit_cont:
        fig_res, ax_res = plt.subplots(figsize=(20, 10))
        off_res = 0
        res_arr = []
        off_res_nd = [0]
        off_lamba = 0 if arm == 'UVB' else -35
        for f_n, s_n, d in zip(f_norm, sigma_norm, date_obs):
            ax_res.plot(lamba_median, f_n - f_median + off_res, lw=0.5)
            ax_res.fill_between(lamba_median, off_res - n_sigma*np.sqrt((s_n**2 + sigma_median**2)),
                                off_res + n_sigma*np.sqrt((s_n**2 + sigma_median**2)), alpha=0.8, color='gray')
            ax_res.annotate(f"{np.round(np.ma.std(f_n - f_median), 3)}", (lamba_median.data[-1] + off_lamba, off_res),)
            ax_res.annotate(f"{d.partition('T')[-1]}", (lamba_median[len(lamba_median)//2], off_res), ha='center',
                            path_effects=[patheffects.withStroke(linewidth=3, foreground='w')])
            if obj_name == 'maria' or obj_name == 'highlands':
                off_res += n_sigma*0.02
            if obj_name == 'darkside':
                off_res += n_sigma*0.1
            res_arr.append(f_n - f_median)
            off_res_nd.append(off_res)

        ax_res.set_ylabel(r'$f_{i} - \overline{f}$', fontsize=14)
        ax_res.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
        if obj_name == 'darkside':
            ax_res.set_ylim(-0.1, 0.5)
        ax_res.set_title(f"{moon_loc} {arm} residue at ${n_sigma}\\sigma$", fontsize=20)

        if obj_name != 'darkside':
            kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(res_arr)
            for f_n, label, off in zip(f_norm, kmeans.labels_, off_res_nd):
                ax_res.annotate(f"{label}", (lamba_median.data[0] + 20, off), ha='center',
                                path_effects=[patheffects.withStroke(linewidth=3, foreground='w')])

        plt.show()



    #if plots_fit_cont:
    #    fig_median, ax_median = plt.subplots(figsize=(25, 5))
    #    ax_median.plot(lamba_median, f_median, lw=0.5)
    #    ax_median.set_yscale('log')
    #    ax_median.set_xscale('log')
    #    ax_median.set_ylabel('Normalized flux', fontsize=14)
    #    ax_median.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
    #    ax_median.set_title(f"{moon_loc} {arm} median flux normalization", fontsize=20)
    #    plt.show()
    if median_region:
        f_region_median = f_median*poly_median
        return f_region_median, lamba_median
    else:
        return f_median, lamba_median


def res(obj_name, arm, n, **kwargs):
    f, sigma, airm, date, q_col, lamba = moon(obj_name, arm, norm=True, n_poly=n, **kwargs)  # clip
    norm_median = np.ma.median(f, axis=0)
    iqr = scipy.stats.iqr(f, axis=0)
    f_not_clipped = moon(obj_name, arm, norm=True, **kwargs)[0]  # no clip
    fig, ax = plt.subplots(figsize=(25, 5))
    ax.plot(lamba[0].data, norm_median)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
    r = np.asarray([flux_norm - norm_median for flux_norm in f_not_clipped])
    fig, ax = plt.subplots(figsize=(25, 5))
    ax.plot(lamba[0], r[0])
    plt.show()
    sigma_r = np.asarray([np.sqrt(s**2 + iqr**2) for s in sigma])
    if obj_name == 'highlands':
        col, row = 3, 6
    if obj_name == 'maria':
        col, row = 4, 4
    if obj_name == 'darkside':
        col, row = 2, 2
    fig, axs = plt.subplots(col, row, figsize=(20, 5), layout='constrained')
    fig.suptitle(obj_name + ' ' + arm, fontsize=30)
    for residue, l, err, ax in zip(r, lamba, sigma_r, fig.axes):
        ax.plot(l, residue, 'k', lw=0.5)
        ax.fill_between(l, -err, err, alpha=0.5)
        ax.set_xlabel(r'$\lambda$ (nm)', fontsize=15)
        ax.set_ylim(-0.1, 0.1)
        ax.grid()

    plt.show()


def rms(obj_name, arm, n, s, mask_lamba_nd, lamba_eff, mask_tell, **kwargs):
    s_to_n = []
    rms_ = []
    for lamba_range, l in zip(mask_lamba_nd, lamba_eff):
        print(f"We are in {l} nm")
        f, lamba = fit_the_cont(obj_name, arm, n, s, n_sigma=None, plots_fit_cont=False, wv_lims=lamba_range,
                                mask_tell_nd=mask_tell, **kwargs)
        f_cont = sigma_clip(f, sigma=s, stdfunc=scipy.stats.iqr)
        #fig, ax = plt.subplots()
        #fig.suptitle(f"$\\lambda = {l:.1f}$", fontsize=20)
        #ax.plot(lamba, f_cont)
        #plt.show()
        rms_i = np.ma.std(f_cont)
        rms_.append(rms_i)
        s_to_n.append(np.ma.median(f_cont)/rms_i)

    #fig_, ax_ = plt.subplots(figsize=(15, 5))
    #fig_.suptitle(f"{arm} in {obj_name}", fontsize=20)
    #ax_.plot(lamba_eff, s_to_n, 'o')
    #ax_.grid()
    #ax_.set_xlabel(r'$\lambda$ (nm)', fontsize=12)
    #ax_.set_ylabel(r'$S/N$', fontsize=12)
    #plt.show()

    return s_to_n, lamba_eff


def rel_mol_col(path, arm):
    hdul = fits.open(path)
    fits_param = hdul[3].data
    if arm == 'VIS':
        h2o = np.array(fits_param[24][1:])
        o2 = np.array(fits_param[25][1:])
        return h2o, o2
    if arm == 'NIR':
        h2o = np.array(fits_param[32][1:])
        co2 = np.array(fits_param[33][1:])
        co = np.array(fits_param[34][1:])
        ch4 = np.array(fits_param[35][1:])
        o2 = np.array(fits_param[36][1:])
        return h2o, co2, co, ch4, o2
        pass
    else:
        print('VIS or NIR (no one selected)')


def plot_atm_abundance(obj_name, arm, **kwargs):
    paths_arr = moon(obj_name, arm, mol_earth_atm=True, **kwargs)
    obs_time = [fits.open(p)[0].header['DATE-OBS'] for p in paths_arr]
    obs_time = [d.partition('T')[-1] for d in obs_time]
    obs_time = pd.to_datetime(obs_time)
    date_form = dates.DateFormatter("%H:%M:%S")
    fig, axes = plt.subplots()
    fig.suptitle(obj_name + ' ' + arm)
    if arm == 'VIS':
        h2o_tuple = np.asarray([rel_mol_col(p, arm)[0] for p in paths_arr])
        o2_tuple = np.asarray([rel_mol_col(p, arm)[1] for p in paths_arr])
        col_h2o, err_col_h2o = np.transpose(h2o_tuple)[0], np.transpose(h2o_tuple)[1]
        col_o2, err_col_o2 = np.transpose(o2_tuple)[0], np.transpose(o2_tuple)[1]
        axes.plot(obs_time, col_h2o, 'o', label=r'$H_{2}O$')
        axes.plot(obs_time, col_o2, 'o', label=r'$O_{2}$')
    if arm == 'NIR':
        h2o_tuple = np.asarray([rel_mol_col(p, arm)[0] for p in paths_arr])
        co2_tuple = np.asarray([rel_mol_col(p, arm)[1] for p in paths_arr])
        co_tuple = np.asarray([rel_mol_col(p, arm)[2] for p in paths_arr])
        ch4_tuple = np.asarray([rel_mol_col(p, arm)[3] for p in paths_arr])
        o2_tuple = np.asarray([rel_mol_col(p, arm)[4] for p in paths_arr])
        col_h2o, err_col_h2o = np.transpose(h2o_tuple)[0], np.transpose(h2o_tuple)[1]
        col_co2, err_col_co2 = np.transpose(co2_tuple)[0], np.transpose(co2_tuple)[1]
        col_co, err_col_co = np.transpose(co_tuple)[0], np.transpose(co_tuple)[1]
        col_ch4, err_col_ch2 = np.transpose(ch4_tuple)[0], np.transpose(ch4_tuple)[1]
        col_o2, err_col_o2 = np.transpose(o2_tuple)[0], np.transpose(o2_tuple)[1]
        print(col_co2)
        axes.plot(obs_time, col_h2o, 'o', label=r'$H_{2}O$')
        axes.plot(obs_time, col_co2, 'o', label=r'$CO_{2}$')
        axes.plot(obs_time, col_co, 'o', label=r'$CO$')
        axes.plot(obs_time, col_ch4, 'o', label=r'$CH_{4}$')
        axes.plot(obs_time, col_o2, 'o', label=r'$O_{2}$')
        pass
    axes.xaxis.set_major_formatter(date_form)
    axes.grid()
    axes.legend()
    axes.set_yscale('log')
    axes.set_ylabel('rel mol col')
    axes.set_xlabel('Time (UTC)')
    plt.show()
    pass





#def airMass_moon(obj_moon_arr, arm):
    #fig, ax = plt.subplots(figsize=(14, 8))

    #airm_arr = []
    #for obj in obj_std_moon:
        #airm = moon(obj_name, arm, dim='1D', counts=False, plot=False, figure=None, idx_arr=None, wv_lims=None,
                    #arm_together=False, mask_tell_nd=None)

'''
def airMass_std(obj_std_arr, arm, moon=False):
    """This function plots the evolution of the air mass of the previous telluric stars
    during the night of the observation of the Moon.
    obj_std_arr could be an array of the names of the telluric stars saved in
    home/yiyo/moon_xshoo_pre_molecfit/ or home/yiyo/moon_xshoo_pre_molecfit/;
    depending on the arm. """
    fig, ax = plt.subplots(figsize=(14, 8))

    for obj in obj_std_arr:
        airm1, date1 = stdTell(obj, arm, dim='1D', counts=False, plot=False, ax=None, first=True)[2:4]
        airm2, date2 = stdTell(obj, arm, dim='1D', counts=False, plot=False, ax=None, first=False)[2:4]
        airm_arr = np.array([airm1, airm2])
        date_arr = np.array([date1, date2])
        times_arr = [date.partition('T')[-1] for date in date_arr]
        times_arr = [pd.to_datetime(time) for time in times_arr]
        ax.plot(times_arr, airm_arr, 'o:')
        ax.annotate(obj, (times_arr[-1], airm_arr[-1]), textcoords='offset points', xytext=(0, 10), ha='center')
    if moon:
        airMass_moon()
    date_form = dates.DateFormatter("%H:%M:%S")
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xlabel('Observation time', fontsize=20)
    ax.set_ylabel('Air mass', fontsize=20)
    ax.grid()
    plt.show()
    pass
'''

#def airMass_moon()

'''
fig, axes = plt.subplots(5, 1, figsize=(20, 20), layout='constrained')
std_tells = ['HD145631', 'HD90027', 'HD80781', '35ERI', 'HD31373']
arms = ['UVB', 'VIS', 'NIR']

# each star
for std, ax in zip(std_tells, fig.axes):
    ax.set_title(std, fontsize=12)
    for a in arms:
        stdTell(std, a, dim='1D', counts=False, plot=True, ax=ax)
    ax.grid()
    #ax.legend()
    ax.set_ylim(0)
ax.set_xlabel(r'$\lambda$ (nm)', fontsize=14)

plt.show()
'''
idx_highlands = np.arange(19)
# telluric mask
mask_VIS = [[635.6, 637.6], [686.54, 696.28], [759.10, 769.76], [927.00, 958.50], [980, 1022.3]]
mask_NIR = [[1065, 1080], [1110, 1150], [1340, 1485], [1790, 2030], [2260, 2290], [2430, 2479]]
mask_all = mask_VIS + mask_NIR

# rms mask
# UVB range starts from 318.2 nm
wv_rms_UVB = [325.0, 339.8, 356.1, 373.5, 393.2, 414.5, 438.8, 466.4, 496.8, 531.0, 556.0]
mask_rms_UVB = [[318.3, 331.5], [332.5, 347.1], [348.3, 364.1], [365.1, 382.5], [383.4, 403.3], [403.9, 425.9],
                [427.2, 451.3], [453, 480.6], [482, 512.8], [514.5, 548], [545, 562.9]]
# VIS range starts from 553.66 nm --> the range 980-1022.3 nm (1001.6 nm) is avoided due to bad behaviour
wv_rms_VIS = [568, 585.9, 607.7, 629.5, 653.8, 682.1, 711.2, 742.6, 777.6, 815.8, 860.2, 904.3, 957.3]
mask_rms_VIS = [[560.6, 574.7], [575.5, 596.2], [595.9, 618.5], [617.7, 642.3], [641.8, 668.1], [667.5, 696.8],
                [696.0, 726.0], [726.6, 759.3], [761.7, 797.4], [796, 837.3], [836.2, 882.5], [880.9, 928.4],
                [946.1, 984.9]]
# NIR range starts from 1054.02 nm
# --> the ranges 1341.3-1413.51 nm (1376.31 nm) and 1803.45-1937.04 nm (1867.86 nm) are avoided due to strong telluric
# absorption
wv_rms_NIR = [1089.58, 1136.96, 1188.64, 1245.24, 1307.5, 1452.78, 1538.23, 1634.38, 1743.33, 2011.54, 2179.17, 2377.28]
mask_rms_NIR = [[1067.35, 1112.76], [1112.77, 1162.22], [1162.22, 1216.28], [1216.28, 1275.61], [1275.61, 1341.02],
                [1413.52, 1494.28], [1494.29, 1584.85], [1584.85, 1687.09], [1687.1, 1803.45], [1937.04, 2092.0],
                [2092.0, 2273.91], [2273.92, 2490.48]]

lamba_eff_nd = [wv_rms_UVB, wv_rms_VIS, wv_rms_NIR]
lamba_rms_mask = [mask_rms_UVB, mask_rms_VIS, mask_rms_NIR]

#fig1, axs1 = plt.subplots(2, 2, figsize=(20, 10), layout='constrained')
#moon('maria', 'NIR', dim='1D', counts=True, plot=True, figure=fig1,
#     wv_lims=None, mask_tell_nd=None, norm=False, arm_together=True)
#plt.show()
#moon('highlands', 'VIS', dim='1D', counts=False, plot=True, figure=fig1,
     #mask_tell_nd=mask_all, norm=False, arm_together=True)
#moon('maria', 'NIR', dim='1D', counts=False, plot=True, figure=fig1,
     #mask_tell_nd=mask_all, norm=False, arm_together=True)
#fig1.suptitle('maria', fontsize=20)
#plt.show()


fig3, axs3 = plt.subplots(1, 1, figsize=(15, 5))
for c, region in zip(['y', 'aqua', 'k'], ['highlands', 'maria', 'darkside']):
    #for arm, rms_mask, lamba_eff in zip(['UVB', 'VIS', 'NIR'], lamba_rms_mask, lamba_eff_nd):
    for arm, rms_mask, lamba_eff in zip(['NIR'], [mask_rms_NIR], [wv_rms_NIR]):
        if region == 'highlands' and arm == 'NIR':
            continue
        idx_arr = np.arange(0, 14) if arm == 'NIR' and region == 'maria' else None
        f, w = fit_the_cont(region, arm, 4, s=1, n_sigma=1, plots_fit_cont=False, median_region=True,
                            mask_tell_nd=mask_all, idx_arr=idx_arr)
        #if arm == 'UVB':
        if arm == 'NIR':
            if region == 'maria':
                lbl = 'Mare Nubium'
            if region == 'highlands':
                lbl = 'Mare Imbrium'
            if region == 'darkside':
                lbl = 'Mare Fecundidatis'
            axs3.plot(w, f, label=lbl, c=c, lw=0.5)
        else:
            axs3.plot(w, f, c=c, lw=0.5)
        # s_to_n, wv_eff = rms('maria', arm, 4, rms_mask, lamba_eff, mask_all, idx_arr=idx_arr)
        # axs3[1].plot(lamba_eff, s_to_n, '+', markersize=8, c='b')
axs3.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
y_label = 'erg ' + r'$cm^{-2} s^{-1} \AA^{-1}$'
axs3.set_ylabel(y_label, fontsize=14)
axs3.set_xscale('log')
axs3.set_yscale('log')
axs3.grid()
axs3.legend()
plt.show()

'''
fig2, axs2 = plt.subplots(2, 2, figsize=(70, 25), layout='constrained')
# for arm in ['UVB', 'VIS']:
for arm in ['VIS']:
    moon('maria', arm, dim='1D', counts=False, plot=True, figure=fig2, idx_arr=None,
         arm_together=True, wv_lims=[600, 650], mask_tell_nd=mask_VIS, norm=True)
fig2.suptitle('Maria one single frame', fontsize=20)
plt.show()
'''

#for arm, mask in zip(['UVB', 'VIS', 'NIR'], [None, mask_VIS, mask_NIR]):
    #res('darkside', arm, mask_tell_nd=mask)

