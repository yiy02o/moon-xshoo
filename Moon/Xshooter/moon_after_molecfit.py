import os
import time
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib import patheffects
import numpy as np
from astropy.io import fits
import astropy.units as u
import scipy
import pandas as pd
from astropy.stats import sigma_clip
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin


def compute_flux(path, dim, mode="post_molecfit"):
    hdul = fits.open(path)
    date_obs = hdul[0].header['DATE-OBS']
    ra_obs, dec_obs = hdul[0].header['RA'], hdul[0].header['DEC']
    pos_obs = [ra_obs, dec_obs]
    ra_off, dec_off = hdul[0].header['HIERARCH ESO SEQ CUMOFF RA'], hdul[0].header['HIERARCH ESO SEQ CUMOFF DEC']
    pos_cumOff = [ra_off, dec_off]
    airm_initial = hdul[0].header['HIERARCH ESO TEL AIRM START']
    airm_final = hdul[0].header['HIERARCH ESO TEL AIRM END']
    airm = (airm_initial + airm_final) / 2
    posang = hdul[0].header['HIERARCH ESO ADA POSANG']
    flux = hdul[0].data.copy() if dim == '1D' else np.median(hdul[0].data.copy(), axis=0)
    err_flux = hdul[1].data.copy() if dim == '1D' else np.std(hdul[0].data.copy(), axis=0)
    if mode == "post_molecfit":
        wv = hdul[4].data['lambda'] * u.micron.to(u.nm)   # to avoid saturation
    hdul.close()
    if mode == "post_molecfit":
        return flux, err_flux, airm, date_obs, pos_obs, pos_cumOff, posang, wv
    else:
        return flux, err_flux, airm, date_obs, pos_obs, pos_cumOff, posang


def idp_subtract(path):
    hdul = fits.open(path)
    bin_table = hdul[1].data[0]
    obj_region = hdul[0].header['HIERARCH ESO OBS NAME']
    quants_col = hdul[1].columns
    return bin_table, obj_region, quants_col


def norm_slope(y, x, n_poly):
    z = np.ma.polyfit(x, y, n_poly)
    p = np.poly1d(z)
    y_poly = p(x.data)
    return y_poly


def mask_tellSpikes(lamba, X_n):
    mask_nd = np.asarray([np.logical_and(lamba > r[0], lamba < r[1]) for r in X_n])
    mask_reduced = np.logical_or.reduce(mask_nd)
    return mask_reduced


def moon(obj_name, xshoo_arm, dim='1D', counts=False, idx_nd=None, wv_lims=None, skip_frame=None, mask_tell_nd=None,
         norm=False, n_poly=4, mol_earth_atm=False, mode="post_molecfit"):
    """The UVB data products (no molecfit correction) are saved in the directory
    /home/yiyo/moon_xshoo_pre_molecfit/. Conversely, VIS and NIR arm must be corrected with molecfit
    and are located in the directory /home/yiyo/moon_xshoo_after_molecfit. Most of the conditionals
    used in the code to search the files are a consequence of how the directories, subdirectories and
    files are sorted. So to find the data, an obj_name (--> 'maria' for example) and an
    arm (--> 'UVB') has to be provided. Also, you must specify if you want to work with the 1D or 2D
    merged spectrum (in the case of 2D spectrum, the median along the slit is computed for the equivalent
    one dimensional flux) and if you want to work with the counts or physical units (boolean).
    """

    if mode == "pre_molecfit" or counts:
        src = '/home/yiyo/moon_xshoo_pre_molecfit/'
        for directory in os.listdir(src):
            if obj_name in directory:
                src_complement = src + directory + '/reflex_end_products/'
                list_obs_files = os.listdir(src_complement)
                if idx_nd is None:
                    idx_files = list_obs_files
                else:
                    idx_files = []
                    for idx in idx_nd:
                        if idx != skip_frame:
                            idx_files.append(list_obs_files[idx])
                        else:
                            continue
                flux_arr_per_obs = []
                err_flux_arr_per_obs = []
                airm_arr_per_obs = []
                date_arr_per_obs = []
                pos_arr_per_obs = []
                poscumOff_arr_per_obs = []
                posang_arr_per_obs = []
                for subdir in idx_files:
                    # each subdir represents one single observation
                    subsubdir_list = os.listdir(src_complement + subdir + '/')
                    for subsubdir in subsubdir_list:
                        # searching in the arm subdirectories
                        if subsubdir == 'README':
                            continue
                        if all(xshoo_arm in word for word in os.listdir(src_complement+subdir+'/'+subsubdir+'/')):
                            for file in os.listdir(src_complement + subdir + '/' + subsubdir + '/'):
                                keyname = 'SLIT_MERGE' if counts else 'SLIT_FLUX_MERGE'
                                if keyname + dim in file:
                                    path_file = os.path.join(src_complement + subdir + '/' + subsubdir + '/', file)
                                    flux, err_flux, airm, date_obs, pos_obs, \
                                        pos_cumOff, posang = compute_flux(path_file, dim, mode=mode)
                                    flux_arr_per_obs.append(flux)
                                    err_flux_arr_per_obs.append(err_flux)
                                    airm_arr_per_obs.append(airm)
                                    date_arr_per_obs.append(date_obs)
                                    pos_arr_per_obs.append(pos_obs)
                                    poscumOff_arr_per_obs.append(pos_cumOff)
                                    posang_arr_per_obs.append(posang)
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
                if idx_nd is None:
                    idx_files = list_obs_files
                else:
                    idx_files = []
                    for idx in idx_nd:
                        if idx != skip_frame:
                            idx_files.append(list_obs_files[idx])
                        else:
                            continue
                    # idx_files = [list_obs_files[idx] for idx in idx_arr]

                bin_table_arr_per_obs = []
                obj_region_arr_per_obs = []
                quants_col_arr_per_obs = []
                for subdir in idx_files:
                    # each subdir represents one single observation
                    for subsubdir in os.listdir(src_complement + subdir + '/'):
                        # searching in the arm subdirectories
                        if subsubdir == 'README':
                            continue
                        if all(xshoo_arm in word for word in
                               os.listdir(src_complement + subdir + '/' + subsubdir + '/')):
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

    else:
        src = '/home/yiyo/moon_xshoo_after_molecfit/'
        for directory in os.listdir(src):
            if obj_name in directory:
                src_complement = src + directory + '/'
                for dim_directory in os.listdir(src_complement):
                    if dim in dim_directory:
                        new_src_complement = src_complement + dim + '/'
                        for arm_dir in os.listdir(new_src_complement):
                            if xshoo_arm in arm_dir:
                                pre_file_src = new_src_complement + arm_dir + '/reflex_end_products/molecfit/XSHOOTER/'
                                list_obs_files = os.listdir(pre_file_src)
                                if idx_nd is None:
                                    idx_files = list_obs_files
                                else:
                                    idx_files = []
                                    for idx in idx_nd:
                                        if idx != skip_frame:
                                            idx_files.append(list_obs_files[idx])
                                        else:
                                            continue
                                flux_arr_per_obs = []
                                err_flux_arr_per_obs = []
                                airm_arr_per_obs = []
                                date_arr_per_obs = []
                                pos_arr_per_obs = []
                                poscumOff_arr_per_obs = []
                                posang_arr_per_obs = []
                                wv_arr_per_obs = []
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
                                                flux, err_flux, airm, date_obs, \
                                                    pos_obs, pos_cumOff, posang, wv = compute_flux(file_src, '1D',
                                                                                                   mode=mode)
                                                flux_arr_per_obs.append(flux)
                                                err_flux_arr_per_obs.append(err_flux)
                                                airm_arr_per_obs.append(airm)
                                                date_arr_per_obs.append(date_obs)
                                                pos_arr_per_obs.append(pos_obs)
                                                poscumOff_arr_per_obs.append(pos_cumOff)
                                                posang_arr_per_obs.append(posang)
                                                wv_arr_per_obs.append(wv)
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
    # avoid the saturation
    mask_flux = []
    if not xshoo_arm == 'UVB' and not xshoo_arm == 'VIS' and not xshoo_arm == 'NIR':
        raise ValueError("Can you repeat the arm?")
    # pre molecfit
    if mode == "pre_molecfit":
        for b in bin_table_arr_per_obs:
            if xshoo_arm == 'UVB':
                m_f = np.ma.where(np.logical_and(b['WAVE'] > 374, b['WAVE'] < 3_000))  # 320 initially
            if xshoo_arm == 'VIS':
                m_f = np.ma.where(np.logical_and(b['WAVE'] > 554, b['WAVE'] < 3_000))  # 550 initially
            if xshoo_arm == 'NIR':
                m_f = np.ma.where(np.logical_and(b['WAVE'] > 1050, b['WAVE'] < 3_000))
            mask_flux.append(m_f)
        wave_length = [b['WAVE'][m_f] for b, m_f in zip(bin_table_arr_per_obs, mask_flux)]

    # post molecfit
    else:
        for wv in wv_arr_per_obs:
            if xshoo_arm == 'UVB':
                m_f = np.ma.where(np.logical_and(wv > 374, wv < 3_000))  # 320 initially
            if xshoo_arm == 'VIS':
                m_f = np.ma.where(np.logical_and(wv > 554, wv < 3_000))  # 550 initially
            if xshoo_arm == 'NIR':
                m_f = np.ma.where(np.logical_and(wv > 1050, wv < 3_000))
            mask_flux.append(m_f)
        wave_length = [wv[m_f] for wv, m_f in zip(wv_arr_per_obs, mask_flux)]
    flux_arr_per_obs = [f[m_f] for f, m_f in zip(flux_arr_per_obs, mask_flux)]
    err_flux_arr_per_obs = [err[m_f] for err, m_f in zip(err_flux_arr_per_obs, mask_flux)]

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

    pos_arr_per_obs = np.asarray(pos_arr_per_obs)
    poscumOff_arr_per_obs = np.asarray(poscumOff_arr_per_obs)
    posang_arr_per_obs = np.asarray(posang_arr_per_obs)
    return flux_arr_per_obs, err_flux_arr_per_obs, airm_arr_per_obs, \
        date_arr_per_obs, pos_arr_per_obs, poscumOff_arr_per_obs, posang_arr_per_obs, wave_length


def lambda_ranges_v2(Ek, lamba, n_poly, s, plots_fit_cont=False):
    poly_k = norm_slope(Ek, lamba, n_poly)
    Ek_norm = Ek/poly_k
    Ek_clip = sigma_clip(Ek_norm, sigma=s, stdfunc=scipy.stats.iqr)
    m_clip = Ek_clip.mask
    lamba = np.ma.masked_array(lamba.data, mask=m_clip)

    if plots_fit_cont:
        fig_cont, ax_cont = plt.subplots()
        fig_cont.suptitle("Median region clip para fitear continuo")
        ax_cont.plot(lamba, Ek_clip, lw=0.5)
        ax_cont.set_yscale('log')
        ax_cont.set_xscale('log')
        ax_cont.set_ylabel('Normalized flux', fontsize=14)
        ax_cont.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
        fig_cont.show()
    return lamba


def fit_the_cont(obj_name, xshoo_arm, n_poly, s, n_sigma, plots_fit_cont=True, median_region=False, kmeans=None, k=None,
                 **kwargs):
    if obj_name == 'maria':
        moon_loc = 'Nubium'
    if obj_name == 'highlands':
        moon_loc = 'Imbrium'
    if obj_name == 'darkside':
        moon_loc = 'Fecundidatis'
    f, sigma, airm, date_obs, q, w = moon(obj_name, xshoo_arm, dim='1D', **kwargs)#[0:4] #mask_tell_nd=mask_all,
                                          #mode="pre_molecfit" if xshoo_arm == "UVB" else "post_molecfit", **kwargs)#[0:4]
    #print(f[0].data)
    #print(w[0].data)
    #print(f[0].data[f[0].mask])
    #print(w[0].data[f[0].mask])
    # lambda_ranges(obj_name, xshoo_arm, n_poly, s, plots_fit_cont, **kwargs)
    lamba_clipped = [lambda_ranges_v2(flux, wv, n_poly, s) for flux, wv in zip(f, w)]
    mask_to_clip = [wv_clip.mask for wv_clip in lamba_clipped]
    Ek_clipped = [np.ma.masked_array(flux.data, mask=m_clip) for flux, m_clip in zip(f, mask_to_clip)]
    poly_k = [norm_slope(f_clip, wv_clip, n_poly) for f_clip, wv_clip in zip(Ek_clipped, lamba_clipped)]
    f_norm = [flux/poly for (flux, poly) in zip(f, poly_k)]
    sigma_norm = [err/poly for (err, poly) in zip(sigma, poly_k)]

    # f_norm = [sigma_clip(flux, sigma=s, stdfunc=scipy.stats.iqr) for flux in f_norm]  # continuo
    mask_norm = [np.ma.getmask(f_n) for f_n in f_norm]
    mask_median = np.all(mask_norm, axis=0)
    f_median = np.ma.masked_where(mask_median, np.ma.median(f_norm, axis=0))
    lamba_median = np.ma.masked_where(mask_median, lamba_clipped[0].data)
    sigma_median = np.ma.masked_where(mask_median, np.ma.std(f_norm, axis=0))
    poly_median = np.ma.masked_where(mask_median, np.ma.median(poly_k, axis=0))

    off_res = 0
    res_arr = [f_n - f_median for f_n in f_norm]
    res_std = np.ma.median([np.ma.std(r) for r in res_arr])
    off_res_nd = [0]

    if plots_fit_cont:
        fig_res, ax_res = plt.subplots(figsize=(20, 10))
        wv_lims = kwargs.get('wv_lims', None)
        std_pos = lamba_median[3*len(lamba_median)//4] if wv_lims is None \
            else wv_lims[-1] - (wv_lims[-1] - wv_lims[0])//4
        date_pos = lamba_median.data[len(lamba_median.data)//2] if wv_lims is None else (wv_lims[-1] + wv_lims[0])//2
        for res, s_n, d in zip(res_arr, sigma_norm, date_obs):
            ax_res.plot(lamba_median, res + off_res, lw=0.5)
            ax_res.fill_between(lamba_median, off_res - n_sigma*np.sqrt((s_n**2 + sigma_median**2)),
                                off_res + n_sigma*np.sqrt((s_n**2 + sigma_median**2)), alpha=0.4, color='gray')
            ax_res.annotate(f"{np.round(np.ma.std(res), 3)}", (std_pos, off_res),
                            ha='center', path_effects=[patheffects.withStroke(linewidth=3, foreground='w')])
            ax_res.annotate(f"{d.partition('T')[-1]}", (date_pos, off_res), ha='center',
                            path_effects=[patheffects.withStroke(linewidth=3, foreground='w')])
            if obj_name == 'maria' or obj_name == 'highlands':
                off_res += n_sigma*0.02
            if obj_name == 'darkside':
                off_res += n_sigma*0.1
            off_res_nd.append(off_res)

        ax_res.set_ylabel(r'$f_{i} - \overline{f}$', fontsize=14)
        ax_res.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
        if obj_name == 'highlands':
            ax_res.set_ylim(-0.04, 0.37)
        if obj_name == 'maria':
            ax_res.set_ylim(-0.04, 0.34)
        if obj_name == 'darkside':
            ax_res.set_ylim(-0.1, 0.5)
        ax_res.set_title(f"{moon_loc} {xshoo_arm} residue at ${n_sigma}\\sigma$, $n={n_poly}$ and " + r"$\sigma_{clip}=$"
                         f"${s}$", fontsize=20)
        if wv_lims is not None:
            ax_res.set_xlim(wv_lims[0], wv_lims[-1])

        label_pos = lamba_median.data[len(lamba_median.data)//4]if wv_lims is None \
            else wv_lims[0] + (wv_lims[-1] - wv_lims[0])//4
        for frame, off in enumerate(off_res_nd[:-1]):
            ax_res.annotate(f"{frame + 1}", (label_pos, off), ha='center',
                            path_effects=[patheffects.withStroke(linewidth=3, foreground='w')])
        plt.show()
    if median_region:
        f_region_median = f_median*poly_median
        return f_region_median, lamba_median, sigma_median, res_std
    return f_median, lamba_median


def classify_moon_zones(Y, X, n_clusters=2, lamba_no_kmeans=None, plot=False):
    if lamba_no_kmeans is not None:
        prev_mask = np.ma.getmask(X)
        no_kmeans_mask = mask_tellSpikes(X, lamba_no_kmeans)
        mask_kmeans = np.logical_or.reduce([prev_mask, no_kmeans_mask])
        Y = [np.ma.masked_array(y.data, mask=mask_kmeans) for y in Y]
        if plot:
            off = 0
            for y in Y:
                plt.plot(X, y + off)
                off += 0.2
            plt.show()
    k_means = KMeans(n_clusters=n_clusters, n_init=1, random_state=0)
    k_means.fit(Y)
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels = pairwise_distances_argmin(Y, k_means_cluster_centers)
    return k_means_labels


def rms(obj_name, xshoo_arm, n_poly, s, mask_lamba_nd, lamba_eff, mask_tell, plot_rms=False, **kwargs):
    s_to_n = []
    rms_ = []
    for lamba_range, l in zip(mask_lamba_nd, lamba_eff):
        print(f"We are in {l} nm")
        f, lamba = fit_the_cont(obj_name, xshoo_arm, n_poly, s, n_sigma=None, plots_fit_cont=False, wv_lims=lamba_range,
                                mask_tell_nd=mask_tell, **kwargs)
        f_cont = sigma_clip(f, sigma=s, stdfunc=scipy.stats.iqr)
        if plot_rms:
            fig, ax = plt.subplots()
            fig.suptitle(f"$\\lambda = {l:.1f}$", fontsize=20)
            ax.plot(lamba, f_cont)
            plt.show()
        rms_i = np.ma.std(f_cont)
        rms_.append(rms_i)
        s_to_n.append(np.ma.median(f_cont)/rms_i)

    if plot_rms:
        fig_, ax_ = plt.subplots(figsize=(15, 5))
        fig_.suptitle(f"{xshoo_arm} in {obj_name}", fontsize=20)
        ax_.plot(lamba_eff, s_to_n, 'o')
        ax_.grid()
        ax_.set_xlabel(r'$\lambda$ (nm)', fontsize=12)
        ax_.set_ylabel(r'$S/N$', fontsize=12)
        plt.show()

    return s_to_n, lamba_eff


def zoom_median_spectra(flux, lamba, wv_lims):
    lower, upper = wv_lims
    mask_lims = ~np.logical_and(lamba >= lower, lamba <= upper)
    flux_with_zoom = np.ma.masked_array(flux, mask_lims)
    lamba_with_zoom = np.ma.masked_array(lamba, mask_lims)
    return flux_with_zoom, lamba_with_zoom


def rel_mol_col(path, xshoo_arm):
    hdul = fits.open(path)
    fits_param = hdul[3].data
    if xshoo_arm == 'VIS':
        h2o = np.array(fits_param[24][1:])
        o2 = np.array(fits_param[25][1:])
        return h2o, o2
    if xshoo_arm == 'NIR':
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


idx_highlands = np.arange(19)
# telluric mask
mask_VIS = [[550, 580], [635.6, 638], [686.54, 696.28], [759.10, 769.76], [852, 857], [927, 1_022.3]]   # [927.00, 958.5], [980, 1022.3]]
mask_NIR = [[1065, 1080], [1110, 1160], [1260, 1275], [1_340, 1_460],  # [1340, 1485],
            [1720, 2030], [2260, 2290], [2360, 3000]]
mask_all = mask_VIS + mask_NIR

# rms mask
# UVB range starts from 374 nm
wv_rms_UVB = [374, 393.2, 414.5, 438.8, 466.4, 496.8, 531.0, 556.0]
mask_rms_UVB = [[374., 382.5], [383.4, 403.3], [403.9, 425.9], [427.2, 451.3], [453, 480.6], [482, 512.8], [514.5, 548],
                [545, 562.9]]
# VIS range starts from 553.66 nm --> the range 980-1022.3 nm (1001.6 nm) is avoided due to bad behaviour
wv_rms_VIS = [568, 585.9, 607.7, 629.5, 653.8, 682.1, 711.2, 742.6, 777.6, 815.8, 860.2, 904.3]
mask_rms_VIS = [[560.6, 574.7], [575.5, 596.2], [595.9, 618.5], [617.7, 642.3], [641.8, 668.1], [667.5, 696.8],
                [696.0, 726.0], [726.6, 759.3], [761.7, 797.4], [796, 837.3], [836.2, 882.5], [880.9, 928.4]]
# NIR range starts from 1054.02 nm
# --> the ranges 1341.3-1413.51 nm (1376.31 nm) and 1803.45-1937.04 nm (1867.86 nm) are avoided due to strong telluric
# absorption
wv_rms_NIR = [1089.58, 1136.96, 1188.64, 1245.24, 1307.5, 1452.78, 1538.23, 1634.38, 1743.33, 2011.54, 2179.17, 2377.28]
mask_rms_NIR = [[1067.35, 1112.76], [1112.77, 1162.22], [1162.22, 1216.28], [1216.28, 1275.61], [1275.61, 1341.02],
                [1413.52, 1494.28], [1494.29, 1584.85], [1584.85, 1687.09], [1687.1, 1803.45], [1937.04, 2092.0],
                [2092.0, 2273.91], [2273.92, 2490.48]]

lamba_eff_nd = [wv_rms_UVB, wv_rms_VIS, wv_rms_NIR]
lamba_rms_mask = [mask_rms_UVB, mask_rms_VIS, mask_rms_NIR]

imb_k = np.array([0, 0, 0, 0, 1, 0,
                  0, 1, 0, 0, 0, 1,
                  1, 1, 1, 0, 1, 1])
nub_k = np.array([0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 1,
                  1, 1, 1, 1])
nub_k_nir = np.array([0, 0, 0, 0,
                      0, 0, 0, 0,
                      0, 0, 0, 1,
                      1, 1, -1, -1])  # no incluimos los dos últimos frames en el NIR


# spectra together median region
"""
n_imbW_uvb = 4
n_imbE_uvb = 4
n_imbW_vis = 6
n_imbE_vis = 6
Imb_median_west = []
Imb_median_east = []
lambaImb_median_west = []
lambaImb_median_east = []
fig_def, ax_def = plt.subplots(1, 1, layout='constrained', figsize=(15, 5))
for arm, n in zip(['UVB', 'VIS'], [[n_imbW_uvb, n_imbE_uvb], [n_imbW_vis, n_imbE_vis]]):
    idx_arr = np.arange(0, 18) if arm == 'UVB' else None
    Imb_west, lamda_west = fit_the_cont('highlands', arm, n_poly=n[0], s=1, n_sigma=1, mask_tell_nd=mask_all,
                                        plots_fit_cont=False, idx_nd=idx_arr, k=0, kmeans=imb_k,
                                        median_region=True,
                                        mode="pre_molecfit" if arm == "UVB" else "post_molecfit")[:2]
    Imb_east, lamda_east = fit_the_cont('highlands', arm, n_poly=n[1], s=1, n_sigma=1, mask_tell_nd=mask_all,
                                        plots_fit_cont=False, idx_nd=idx_arr, k=1, kmeans=imb_k,
                                        median_region=True,
                                        mode="pre_molecfit" if arm == "UVB" else "post_molecfit")[:2]
    Imb_median_west.append(Imb_west)
    Imb_median_east.append(Imb_east)
    lambaImb_median_west.append(lamda_west)
    lambaImb_median_east.append(lamda_east)

    if arm == 'UVB':
        # ax_def.plot(lamda_west, Imb_west, lw=0.5, label='West Mare Imbrium', c='y')
        # ax_def.plot(lamda_east, Imb_east, lw=0.5, label='East Mare Imbrium', c='b')
        ax_def.plot(lamda_west, Imb_west, lw=0.5, label='West tycho crater', c='y')
        ax_def.plot(lamda_east, Imb_east, lw=0.5, label='East tycho crater', c='b')
    else:
        ax_def.plot(lamda_west, Imb_west, lw=0.5, c='y')
        ax_def.plot(lamda_east, Imb_east, lw=0.5, c='b')

n_nubW_uvb = 5
n_nubE_uvb = 5
n_nubW_vis = 6
n_nubE_vis = 6
Nub_median_west = []
Nub_median_east = []
lambaNub_median_west = []
lambaNub_median_east = []
for arm, n in zip(['UVB', 'VIS'], [[n_nubW_uvb, n_nubE_uvb], [n_nubW_vis, n_nubE_vis]]):
    Nub_west, lamda_west = fit_the_cont('maria', arm, n_poly=n[0], s=1, n_sigma=1, mask_tell_nd=mask_all,
                                        plots_fit_cont=False, k=0, kmeans=nub_k,
                                        median_region=True,
                                        mode="pre_molecfit" if arm == "UVB" else "post_molecfit")[:2]
    Nub_east, lamda_east = fit_the_cont('maria', arm, n_poly=n[1], s=1, n_sigma=1, mask_tell_nd=mask_all,
                                        plots_fit_cont=False, k=1, kmeans=nub_k,
                                        median_region=True,
                                        mode="pre_molecfit" if arm == "UVB" else "post_molecfit")[:2]
    Nub_median_west.append(Nub_west)
    Nub_median_east.append(Nub_east)
    lambaNub_median_west.append(lamda_west)
    lambaNub_median_east.append(lamda_east)
    ax_def.plot(lamda_west, Nub_west, lw=0.5, c='cyan')
    ax_def.plot(lamda_east, Nub_east, lw=0.5, c='orange')

n_nubW_nir = 6
n_nubE_nir = 6
Nub_west_nir, lamda_west_nir = fit_the_cont('maria', 'NIR', n_poly=n_nubW_nir, s=1, n_sigma=1, mask_tell_nd=mask_all,
                                            plots_fit_cont=False, k=0, kmeans=nub_k_nir,
                                            median_region=True)[:2]
Nub_east_nir, lamda_east_nir = fit_the_cont('maria', 'NIR', n_poly=n_nubE_nir, s=1, n_sigma=1, mask_tell_nd=mask_all,
                                            plots_fit_cont=False, k=1, kmeans=nub_k_nir,
                                            median_region=True)[:2]
Nub_median_west.append(Nub_west_nir)
Nub_median_east.append(Nub_east_nir)
lambaNub_median_west.append(lamda_west_nir)
lambaNub_median_east.append(lamda_east_nir)
#ax_def.plot(lamda_west_nir, Nub_west_nir, lw=0.5, label='West Mare Nubium', c='cyan')
#ax_def.plot(lamda_east_nir, Nub_east_nir, lw=0.5, label='East Mare Nubium', c='orange')
ax_def.plot(lamda_west_nir, Nub_west_nir, lw=0.5, label='West Mare Imbrium', c='cyan')
ax_def.plot(lamda_east_nir, Nub_east_nir, lw=0.5, label='East Mare Imbrium', c='orange')

n_fec_uvb = 4
n_fec_vis = 6
n_fec_nir = 4
Fec_median = []
lambaFec_median = []
for arm, n in zip(['UVB', 'VIS', 'NIR'], [n_fec_uvb, n_fec_vis, n_fec_nir]):
    Fec, lamda_i = fit_the_cont('darkside', arm, n_poly=n, s=1, n_sigma=1, mask_tell_nd=mask_all,
                                plots_fit_cont=False, median_region=True,
                                mode="pre_molecfit" if arm == "UVB" else "post_molecfit")[:2]
    Fec_median.append(Fec)
    lambaFec_median.append(lamda_i)
    if arm == 'UVB':
        ax_def.plot(lamda_i, Fec, lw=0.5, label='Mare Fecundidatis', c='k')
    else:
        ax_def.plot(lamda_i, Fec, lw=0.5, c='k')
ax_def.legend()
ax_def.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
y_label = 'erg ' + r'$cm^{-2} s^{-1} \AA^{-1}$'
ax_def.set_ylabel(y_label, fontsize=14)
ax_def.set_xscale('log')
ax_def.set_yscale('log')
ax_def.grid()
plt.show()
"""

