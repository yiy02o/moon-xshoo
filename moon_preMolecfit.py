import os, shutil
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from astropy.io import fits


def plot_default(directory, plot=False, verbose=False):
    for file in os.listdir(directory):
        #if file.endswith('IDP_' + fq_range + '.fits'):
        if 'IDP' in file:
            f = os.path.join(directory, file)
            hdul = fits.open(f)
            obj_region = hdul[0].header['HIERARCH ESO OBS NAME']
            fq_range = hdul[0].header['DISPELEM']
            bin_table = hdul[1].data[0]
            quantts_col = hdul[1].columns
            if plot:
                wavelength = bin_table['WAVE']
                flux = bin_table['FLUX_REDUCED']
                #flux = bin_table['FLUX'][1500:]
                err = bin_table['ERR_REDUCED']
                fig, axs = plt.subplots(figsize=(8, 6))

                # plt.clf()
                # axs.plot(wavelength, flux_reduced)
                axs.plot(wavelength, flux)
                axs.fill_between(wavelength, flux - err, flux + err, alpha=0.3)
                axs.set_title(obj_region + ' at ' + fq_range + ' range')
                axs.set_xlabel('Wavelenght [' + quantts_col['WAVE'].unit + ']')
                axs.set_ylabel('F [' + quantts_col['FLUX_REDUCED'].unit + ']')
                axs.grid()
                plt.show()
                break
            else:
                return bin_table


def median_2DSLIT(directory, ARM, normalize=False, plot=False, axes=None):
    """Calcula al mediana a lo largo del ancho de una slit en función de la frecuencia. Es una cantidad
    por pointing."""
    for folder in os.listdir(directory):
        if folder == 'README':
            pass
        else:
            for file in os.listdir(directory + '/' + folder):
                if 'SCI_SLIT_FLUX_MERGE2D_' + ARM in file:
                    f = os.path.join(directory + '/' + folder, file)
                    hdul = fits.open(f)
                    date_obs = hdul[0].header['DATE-OBS']
                    flux2D = hdul[0].data.copy()
                    median_flux = np.median(flux2D, axis=0)
                    std_flux = np.std(flux2D, axis=0)
                    hdul.close()

                if 'IDP_' + ARM in file:  # the info of our data
                    f = os.path.join(directory + '/' + folder, file)
                    hdul = fits.open(f)
                    bin_table = hdul[1].data[0]
                    obj_region = hdul[0].header['HIERARCH ESO OBS NAME']
                    quants_col = hdul[1].columns
    idx_avoid = 700
    # normalize range
    if ARM == 'NIR':
        idx_min, idx_max = -21_000, -19_000
    if ARM == 'VIS':
        idx_min, idx_max = -10_000, -8_000
    if ARM == 'UVB':
        idx_min, idx_max = -5_000, -3_000
    if normalize:
        flux_test = median_flux[idx_min:idx_max + 1]
        median_test_flux = np.median(flux_test)
        flux_norm = median_flux/median_test_flux
        std_norm = std_flux/median_test_flux
        if plot:
            wv = bin_table['WAVE']
            w_min, w_max = wv[idx_min], wv[idx_max]
            axes.plot(wv[idx_avoid:], flux_norm[idx_avoid:], lw=0.5, label=date_obs, color='b')
            axes.axvspan(w_min, w_max, alpha=0.1, color='k')
            #axes.fill_between(wv[idx_avoid:], flux_norm[idx_avoid:] - std_norm[idx_avoid:],
                              #flux_norm[idx_avoid:] + std_norm[idx_avoid:], alpha=0.3, color='k')
            axes.set_xlabel('[' + quants_col['WAVE'].unit + ']', fontsize=11)
            axes.set_ylabel('Normalized flux', fontsize=11)
            axes.grid()
            axes.legend(fontsize=13)
        else:
            return flux_norm, bin_table, obj_region, quants_col

    else:
        if plot:
            wv = bin_table['WAVE']
            axes.plot(wv[idx_avoid:], median_flux[idx_avoid:], lw=0.5, label=date_obs)
            axes.set_xlabel('[' + quants_col['WAVE'].unit + ']', fontsize=11)
            axes.set_ylabel('erg ' + r'$cm^{-2} s^{-1} \AA^{-1}$', fontsize=11)
            axes.grid()
            axes.legend(fontsize=13)
        else:
            return median_flux, bin_table, obj_region, quants_col


def medianALL_per_Region(directory, ARM, axes, color='b', normalize=False, plot=True):
    """Plotea y calcula la mediana de todos los espectros en las 3 bandas, que se tomaron en cierta zona del objeto.
    Es una cantidad por región del objeto."""
    fluxes = []
    wvs_list = []
    for run_folder in os.listdir(directory):
        flux, bin_table, loc, col = median_2DSLIT(directory + run_folder, ARM, normalize=normalize)
        fluxes.append(flux)
        wvs_list.append(bin_table['WAVE'])
        region = loc
        quants_columns = col
    if len(fluxes) == 19:   # We discard the last bad data of highlands pointing
        fluxes = fluxes[:-1]
    if len(fluxes) == 16 and ARM == 'NIR':
        fluxes = fluxes[:-2]
    print(len(fluxes))
    median_ALL = np.median(np.asarray(fluxes), axis=0)
    std_ALL = np.std(np.asarray(fluxes), axis=0)
    idx_avoid = 700
    if plot:
        axes.plot(wvs_list[0][idx_avoid:], median_ALL[idx_avoid:], linewidth=0.5, color=color, label=region)
        axes.axhline(1., color='grey', ls='--', lw=3, alpha=0.8)
        y_label = r'$I$' + quants_columns['FLUX'].unit + ']'
        if normalize:
            y_label = 'Flux normalized'
        axes.set_ylabel(y_label, fontsize=13)
        axes.set_xlabel('[' + quants_columns['WAVE'].unit + ']', fontsize=13)
        axes.set_title(ARM, fontsize=16)  # --> to plot per arm joint regions
        axes.tick_params(axis='both', labelsize=12)
        axes.legend(fontsize=13)
    else:
        return np.array([np.asarray(wvs_list[0][idx_avoid:]), median_ALL[idx_avoid:], std_ALL[idx_avoid:]])


def plot_medianZone_together(path_arr, normalize=False, plot=True):
    """ Plotea las mediana del espectro por zona (tres espectros promedios -> HL, Maria y Darkside)
    """
    ARMS = ['UVB', 'VIS', 'NIR']
    if plot:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 10), layout="constrained")
        for arm, ax in zip(ARMS, axs):
            ax.grid()
            for c, path in zip(['b', 'r', 'k'], path_arr):
                medianALL_per_Region(path, arm, ax, color=c, normalize=normalize, plot=plot)
        plt.show()
    else:
        ax = None
        per_band = []
        wave_per_band = []
        for idx, arm in enumerate(ARMS):
            per_region = np.asarray([medianALL_per_Region(path, arm, ax, normalize=normalize, plot=plot)[1] for path
                                     in path_arr])
            wave_per_region = np.asarray([medianALL_per_Region(path, arm, ax, normalize=normalize, plot=plot)[0] for
                                          path in path_arr])
            per_band.append(per_region)
            wave_per_band.append(wave_per_region[0])
        return wave_per_band, per_band


def residues(path_arr, plot=False, norm_with_res=False):
    """Calcula los resiuduos de flujos normalizados entre dos zonas"""
    wave_per_band = []
    F0 = []
    F1 = []
    res_per_band = []
    std_per_band = []
    ax = None
    ARMS = ['UVB', 'VIS', 'NIR']
    for arm in ARMS:
        std_per_region = np.asarray([medianALL_per_Region(path, arm, ax, normalize=True, plot=False)[2] for path
                                     in path_arr])
        flux_per_region = np.asarray([medianALL_per_Region(path, arm, ax, normalize=True, plot=False)[1] for path
                                      in path_arr])
        wave_per_region = np.asarray([medianALL_per_Region(path, arm, ax, normalize=True, plot=False)[0] for path
                                      in path_arr])
        flux_0 = flux_per_region[0]
        F0.append(flux_0)
        flux_1 = flux_per_region[1]
        F1.append(flux_1)
        res = flux_0 - flux_1
        res_per_band.append(res)
        wave_per_band.append(wave_per_region[0])
        std_per_band.append(std_per_region)
    if plot:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 10), layout="constrained")
        zone_0 = path_arr[0].partition('_')[-1].partition('_')[-1].partition('/')[0]
        zone_1 = path_arr[1].partition('_')[-1].partition('_')[-1].partition('/')[0]
        fig.suptitle(zone_0 + ' - ' + zone_1, fontsize=35)
        for w, res, f0, f1, std, arm, ax in zip(wave_per_band, res_per_band, F0, F1, std_per_band, ARMS, axs):
            ax.plot(w, res, lw=0.3, color='k', alpha=1)
            if norm_with_res:
                ax.plot(w, f0, lw=0.3, color='b', label=zone_0)
                ax.plot(w, f1, lw=0.3, color='orange', label=zone_1)
                ax.legend()
            sigma_i = np.sqrt((std[0]**2+std[1]**2))
            ax.fill_between(w, -sigma_i, sigma_i, alpha=0.1, color='k')
            ax.set_xlabel('[nm]', fontsize=13)
            ax.set_ylabel('Res', fontsize=13)
            ax.grid()
            ax.set_title(arm, fontsize=13)
            ax.tick_params(axis='both', labelsize=12)
        plt.show()
    else:
        return wave_per_band, res_per_band, std_per_band


# demo data with default oca rules mode
# path_demo = '/home/yiyo/reflex_data/reflex_end_products/2023-04-30T03:28:15/'


path_demo = '/home/yiyo/reflex_data/reflex_end_products/2023-04-30T04:26:13/'


# MOON highlands, 1 single frame with 1 single sky reduced with the new mapping oca rule (-> 3 files)
path_Highlands = '/home/yiyo/reflexData_mapMode_Highlands/reflex_end_products/'
path_Maria = '/home/yiyo/reflexData_mapMode_Maria/reflex_end_products/'
path_dark_side = '/home/yiyo/reflexData_mapMode_Darkside/reflex_end_products/'


path_arrays = [path_Highlands, path_Maria, path_dark_side]

n_subdir = len(os.listdir(path_Highlands))

fig, axs = plt.subplots(2, 2, figsize=(30, 15), layout="constrained")
path_0 = path_dark_side
arm_0 = 'NIR'
for f, ax in zip(os.listdir(path_0), fig.axes):
    median_2DSLIT(path_0 + f, arm_0, normalize=True, plot=True, axes=ax)
    fig.suptitle(path_0.partition('_')[-1].partition('/')[0].partition('_')[-1] + ' {}'.format(arm_0), fontsize=15)
plt.show()


'''
# Plotea las mediana del espectro por zona (tres espectros promedios -> HL, Maria y Darkside)
ARMS = ['UVB', 'VIS', 'NIR']
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(13, 10), layout="constrained")
for arm, ax in zip(ARMS, axs):
    ax.grid()
    for c, path in zip(['k', 'r'], [path_Highlands, path_Maria]):  # to plot
        plot_medianALL_per_Region(path, arm, color=c, axes=ax)
#ax.set_yscale('log')
plt.show()'''