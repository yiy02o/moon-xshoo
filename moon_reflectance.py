import pandas
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from moon_after_molecfit import *
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
from PyAstronomy import pyasl
import astropy.units as u
import astropy.constants as C
from brokenaxes import brokenaxes


def lambda_masked(nd_arr, condition):
    mask = mask_tellSpikes(nd_arr, condition)
    masked_data = np.ma.masked_array(nd_arr, mask)
    return masked_data


def geometricAlbedo(E_nu, E_s_nu, alpha, slit_width, d_S_B=0.99386972*u.AU, d_solar_twin=70*u.pc,
                    d_E_B=399_184.62*u.km, normalize=None, lamba=None):
    r_earth = 6378.14*u.km
    r_moon = 1737*u.km
    d_E_B_mean = 384_400*u.km
    a = slit_width*r_moon/900   # subtended Moon radius 900''
    b = 11*r_moon/900           # 11'' is the slit height of X-Shooter
    Omega = a*b/(d_E_B - r_earth)**2
    f_d = (d_S_B/d_solar_twin.to(u.AU))**2 * (d_E_B/d_E_B_mean)**2
    E_ratio = E_nu/E_s_nu
    phi_alpha = (np.sin(alpha*np.pi/180) + (np.pi - alpha*np.pi/180)*np.cos(alpha*np.pi/180))/np.pi
    output = E_ratio * f_d * np.pi / (Omega * phi_alpha)
    return output


def plot_zone_star_alb(fig, ax, lamba, E_nu, lamba_solar, E_s_nu, n_poly, alpha, slit_width, reg, lamba_band,
                       title=None, color='k', zoomx=None, norm_moon=None, norm_alb=None, lw=0.5, alpha_c=0.6, ls='-',
                       **kwargs):
    y_label = 'erg ' + r'$cm^{-2} s^{-1} \AA^{-1}$'
    if lamba_band == 'UVB':
        ax[2].set_ylabel(r'$A_g$')
        ax[2].set_xlabel(r'$\lambda$ (nm)')
        ax[2].grid()
        ax[0].set_ylabel(y_label)
        ax[0].set_xlabel(r'$\lambda_{rest}$ (nm)')
        ax[0].grid()
        ax[1].set_ylabel(y_label)
        ax[1].set_xlabel(r'$\lambda_{obs} = (1 + \Delta v / c)\lambda_{rest}$ (nm)')
        ax[1].grid()
    # Uncorrected albedo
    A_g_k = oneForAll(lamba_solar, E_s_nu, lamba, E_nu, n_poly, alpha, slit_width, ds_corr=False, **kwargs)

    # Doppler shift
    A_g_k_shifted, rv = oneForAll(lamba_solar, E_s_nu, lamba, E_nu, n_poly, alpha, slit_width, **kwargs)[0:-1]

    if zoomx is not None:
        lower, upper = zoomx
        mask_before = lamba.mask
        mask_s_before = lamba_solar.mask
        if lamba_band == 'UVB':
            mask_before = np.zeros(len(lamba), dtype=bool)
            mask_s_before = np.zeros(len(lamba_solar), dtype=bool)
        mask_lim = np.logical_or.reduce([~np.logical_and(lamba >= lower, lamba <= upper), mask_before], axis=0)
        mask_lim_s = np.logical_or.reduce([~np.logical_and(lamba_solar >= lower, lamba_solar <= upper), mask_s_before],
                                          axis=0)
        lamba = np.ma.masked_array(lamba.data, mask_lim)
        lamba_solar = np.ma.masked_array(lamba_solar.data, mask_lim_s)
        A_g_k = np.ma.masked_array(A_g_k.data, mask_lim)
        A_g_k_shifted = np.ma.masked_array(A_g_k_shifted.data, mask_lim)
        E_nu = np.ma.masked_array(E_nu.data, mask_lim)
        E_s_nu = np.ma.masked_array(E_s_nu.data, mask_lim_s)
    if np.all(A_g_k_shifted.mask):
        return print(f'{lamba_band} arm outside of the zoom region')
    if norm_alb is not None:
        A_g_k = A_g_k/norm_alb
    #if lamba_band == 'UVB':
        #ax[2].plot(lamba, A_g_k, lw=lw, c=color, alpha=alpha_c, ls=ls, label='Uncorrected albedo')
    #else:
        #ax[2].plot(lamba, A_g_k, lw=lw, c=color, alpha=alpha_c, ls=ls)
    if lamba_band == 'NIR':
        ax[2].plot(lamba, A_g_k_shifted, lw=lw, alpha=alpha_c, ls=ls,
                   label=f"Corrected albedo")
    else:
        ax[2].plot(lamba, A_g_k_shifted, lw=lw, alpha=alpha_c, ls=ls)
    if lamba_band == 'UVB':
        ax[2].legend()
    if norm_moon is not None:
        E_nu = E_nu/norm_moon
    if lamba_band == 'UVB':
        ax[0].plot(lamba, E_nu, lw=lw, c=color, label=reg, alpha=alpha_c, ls=ls)
    else:
        ax[0].plot(lamba, E_nu, lw=lw, c=color, alpha=alpha_c, ls=ls)
    if lamba_band == 'NIR':
        ax[0].legend()
    ax[1].plot(lamba_solar, E_s_nu, lw=lw, c='k', ls=ls)
    if zoomx is not None:
        ax[1].plot(lamba_solar/(1 + rv/C.c.to(u.km/u.s).value), E_s_nu, lw=lw, c='r', ls='dashed')
    if title is not None:
        fig.suptitle(title, fontsize=40)
    pass


def normalizeXXnm(y_arr, lamba, norm_range):
    mask_range = ~np.logical_and(lamba.data > norm_range[0], lamba.data < norm_range[1])
    mask_from_before = y_arr.mask
    new_mask = np.logical_or.reduce(np.asarray([mask_range, mask_from_before]))
    y_arr_masked = np.ma.masked_array(y_arr.data, mask=new_mask)
    # lamba_masked = np.ma.masked_array(lamba.data, mask=new_mask)
    # fig, ax = plt.subplots()
    # ax.plot(lamba_masked, y_arr_masked)
    # ax.plot(lamba_masked, np.ones(len(lamba_masked))*np.ma.median(y_arr_masked))
    # plt.show()
    median_to_norm = np.ma.median(y_arr_masked)
    return median_to_norm


def compute_dRV(w, f, tw, tf, rvmin, rvmax, drv, skipedge=20, plot=False):
    if plot:
        plt.title('Template (blue) and spectra shifted (red), both normalized, before RV correction')
        plt.plot(tw, tf, 'b.-')
        plt.plot(w, f, 'r.-')
        plt.grid()
        plt.show()

    rv, cc = pyasl.crosscorrRV(w, f, tw, tf, rvmin, rvmax, drv, skipedge=skipedge)
    maxind = np.argmax(cc)
    print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
    if rv[maxind] > 0:
        print(" A red-shift with respect to the template")
    else:
        print(" A blue-shift with respect to the template")
    if plot:
        plt.plot(rv, cc, 'bp-')
        plt.plot(rv[maxind], cc[maxind], 'ro')
        plt.show()

        plt.title('Template (blue) and spectra shifted (red), both normalized, after RV correction')
        plt.plot(tw, tf, 'b.-')
        plt.plot(w/(1 + rv[maxind]/C.c.to(u.km/u.s).value), f, 'r.-')
        plt.grid()
        plt.show()
    return rv[maxind]


def lambda_ranges_v2(Ek, lamba, n_poly, s, plots_fit_cont=False):
    poly_k = norm_slope(Ek, lamba, n_poly)
    Ek_norm = Ek/poly_k
    Ek_clip = sigma_clip(Ek_norm, sigma=s, stdfunc=scipy.stats.iqr)
    m_clip = Ek_clip.mask
    lamba = np.ma.masked_array(lamba.data, mask=m_clip)

    if plots_fit_cont:
        plt.title("Median region clip para fitear continuo")
        plt.plot(lamba, Ek_clip, lw=0.5)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Normalized flux', fontsize=14)
        plt.xlabel(r'$\lambda$ (nm)', fontsize=14)
        plt.show()
    return lamba


def fit_cont_v2(Ek, lamba, n_poly, s, plot_cont=False):
    lamba_clipped = lambda_ranges_v2(Ek, lamba, 4, s, plots_fit_cont=plot_cont)
    m_clip = lamba_clipped.mask
    Ek_clipped = np.ma.masked_array(Ek.data, mask=m_clip)
    poly_k = norm_slope(Ek_clipped, lamba_clipped, n_poly)
    Ek_norm = Ek/poly_k
    if plot_cont:
        plt.title("Median region continuom fit and the polynomial")
        plt.plot(lamba, Ek, lw=0.5)
        plt.plot(lamba, poly_k, lw=0.5, c='r')
        plt.yscale('log')
        plt.xlabel(r'$\lambda$ (nm)', fontsize=14)
        plt.show()

        plt.title("Median region continuom fit normalized")
        plt.plot(lamba, Ek_norm, lw=0.5)
        plt.yscale('log')
        plt.ylabel('Normalized flux', fontsize=14)
        plt.xlabel(r'$\lambda$ (nm)', fontsize=14)
        plt.show()
    return Ek_norm


def do_GF():
    pass


def search_maskedArea():
    pass


def ma_interp(newx, x, y, mask, propagate_mask=True):
    newy = np.interp(newx, x[~mask], y[~mask])
    if propagate_mask:
        newmask = mask[:]
        newmask[mask] = 1; newmask[~mask] = 0
        newmask = np.interp(newx, x, newmask)
        newy = np.ma.masked_array(newy, newmask > 0.5)
    return newy


def oneForAll(w, f, tw, tf, n_poly, alpha, slit_width, n_sigma=1, rv=None, select=None, rvmin=-50, rvmax=50,
              drv=0.1, skipedge=20, plot=False, ds_corr=True, **kwargs):
    """The objects w (wavelength), f (flux), tw (template wavelength) and tf (template flux) should be
    masked before to avoid strong telluric absorption zones, basically to not corrupt the polynomial continuum fit.
    n_poly (int) and n_sigma (int) are the polynomial degree and sigma level of the polynomial fit and sigma_clipping
    respectively. If you want, the fit process can be shown with plot_cont=True. """
    # In case you don't want to correct for doppler shift
    if not ds_corr:
        return geometricAlbedo(tf, f, alpha, slit_width, **kwargs)
    f_normalized = fit_cont_v2(f, w, n_poly, n_sigma, plot_cont=plot)
    tf_normalized = fit_cont_v2(tf, tw, n_poly, n_sigma, plot_cont=plot)

    # RV compute
    if rv is None:
        if select is None:
            raise ValueError(f"You should provide a wv range (CCOR) or a wv array (GF)")
        if isinstance(select[0], (list, tuple)):
            idx_masked = [np.where((tw.data > s[0]) & (tw.data < s[-1])) for s in select]
            rv = np.mean([compute_dRV(w.data[m], f_normalized.data[m], tw.data[m], tf_normalized.data[m], rvmin, rvmax,
                                      drv, skipedge=skipedge, plot=plot) for m in idx_masked])
            #low_lim = s[0]
            #upper_lim = s[-1]
            #rv_i = compute_dRV(w.data[low_lim:upper_lim], f_normalized.data[low_lim:upper_lim],
                               #tw.data[low_lim:upper_lim], tf_normalized.data[low_lim:upper_lim],
                               #rvmin, rvmax, drv, skipedge=skipedge, plot=plot)
        #else:
            #rv = do_GF()
            #pass
    print(f"The mean rv is {rv}")
    w_corr = tw.data/(1 + rv/C.c.to(u.km/u.s).value)
    spl = UnivariateSpline(w_corr, f.data, s=0)
    data_interp = spl(tw)
    spl_norm = UnivariateSpline(w_corr, f_normalized.data, s=0)
    data_norm_interp = spl_norm(tw)
    f_interp = np.ma.masked_array(data_interp, mask=tf.mask)
    f_normalized_interp = np.ma.masked_array(data_norm_interp, mask=tf.mask)
    if plot:
        plt.plot(tw[5900:6000], tf_normalized[5900:6000], 'b.-', label='Template data')
        plt.plot(w_corr[5900:6000], f_normalized_interp[5900:6000], 'r.-', label='Interpolated data')
        plt.legend()
        plt.grid()
        plt.xlabel(f"$\lambda$ (nm)")
        plt.ylabel(f"Normalized flux")
        plt.show()
    return geometricAlbedo(tf, f_interp, alpha, slit_width, **kwargs), rv, [tf_normalized, f_normalized]


def data_interpolate(w, f, s=0):
    spl = UnivariateSpline(w.data, f.data, s=s)
    return spl(w)


def plot_brokenxaxis(nrow, ncol, selectX, X_arr, Y_arr, n_poly, alpha, slit_width, **kwargs):
    fig = plt.figure(figsize=(20, 4), layout='constrained')
    for sps, select, X, Y in zip(GridSpec(nrow, ncol, figure=fig), selectX, X_arr, Y_arr):
        bax = brokenaxes(xlims=(select), subplot_spec=sps, width_ratios=[1, 1, 1, 1, 1], despine=False, d=0.,
                         wspace=0.01)
        x1, x2 = X
        Y1, Y2 = Y   # y1 and y2 would be the object and solar twin data respectively
        A_g, rv, Y_norm = oneForAll(x2, Y2, x1, Y1, n_poly, alpha, slit_width, **kwargs)
        y1, y2 = Y_norm
        bax.plot(x1, y1, 'b.-', label='Moon')
        bax.plot(x2/(1 + rv/C.c.to(u.km/u.s).value), y2, 'r.-', label='Twin Solar')
        bax.plot([588.38, 588.38], [0, 1.2], ls='--', c='grey')
        bax.plot([588.99, 588.99], [0, 1.2], ls='--', c='grey')
        bax.plot([589.59, 589.59], [0, 1.2], ls='--', c='grey')
        #bax.plot(x1, 10*A_g, color='cyan', label='Geometric albedo')
        bax.grid()
        bax.legend(loc='best', bbox_to_anchor=(0.9, 0.5))
        bax1 = bax.twinx()
        print(bax1)
        for ax in bax1:
            ax.plot(x1, A_g, color='cyan', label='Geometric albedo')

    plt.show()
    pass



path_SolarType_UVB = '/home/yiyo/moon_xshoo_pre_molecfit/reflexData_stdTell_solarType/reflex_end_products/' \
                     '2023-08-17T23:54:12/first/XSHOO.2018-03-10T10:02:47.844_tpl/' \
                     'Hip098197_TELL_SLIT_FLUX_MERGE1D_UVB.fits'
path_SolarType_IDP_UVB = '/home/yiyo/moon_xshoo_pre_molecfit/reflexData_stdTell_solarType/reflex_end_products/' \
                         '2023-08-17T23:54:12/first/XSHOO.2018-03-10T10:02:47.844_tpl/' \
                         'Hip098197_TELL_SLIT_FLUX_IDP_UVB.fits'
path_SolarType_VIS = '/home/yiyo/moon_xshoo_after_molecfit/stdTell_solarType/reflex_end_products/molecfit/XSHOOTER/' \
                     '2023-08-18T02:19:22/Hip098197_TELL_SLIT_FLUX_MERGE1D_VIS/Hip098197_SCIENCE_TELLURIC_CORR.fits'
path_SolarType_IDP_VIS = '/home/yiyo/moon_xshoo_pre_molecfit/reflexData_stdTell_solarType/reflex_end_products/' \
                         '2023-08-17T23:54:12/first/XSHOO.2018-03-10T10:02:53.034_tpl/' \
                         'Hip098197_TELL_SLIT_FLUX_IDP_VIS.fits'
path_SolarType_NIR = '/home/yiyo/moon_xshoo_after_molecfit/stdTell_solarType/reflex_end_products/molecfit/XSHOOTER/' \
                 '2023-08-18T02:19:22/Hip098197_TELL_SLIT_FLUX_MERGE1D_NIR/Hip098197_SCIENCE_TELLURIC_CORR.fits'
path_SolarType_IDP_NIR = '/home/yiyo/moon_xshoo_pre_molecfit/reflexData_stdTell_solarType/reflex_end_products/' \
                         '2023-08-17T23:54:12/first/XSHOO.2018-03-10T10:02:56.137_tpl/' \
                         'Hip098197_TELL_SLIT_FLUX_IDP_NIR.fits'

lamba_solar_UVB = lambda_masked(fits.open(path_SolarType_IDP_UVB)[1].data[0]['WAVE'][1_000:], mask_all)
lamba_solar_VIS = lambda_masked(fits.open(path_SolarType_IDP_VIS)[1].data[0]['WAVE'][1_000:], mask_all)
lamba_solar_NIR = lambda_masked(fits.open(path_SolarType_IDP_NIR)[1].data[0]['WAVE'][1_000:], mask_all)
# lamba_solar_VIS = lambda_masked(fits.open(path_SolarType_VIS)[4].data['lambda'][1_000:]*1e3, mask_all)
# lamba_solar_NIR = lambda_masked(fits.open(path_SolarType_NIR)[4].data['lambda'][1_000:]*1e3, mask_all)
lamba_solar_list = [lamba_solar_UVB, lamba_solar_VIS, lamba_solar_NIR]
lamba_solar_all = np.ma.concatenate((lamba_solar_UVB, lamba_solar_VIS, lamba_solar_NIR))
Js_tell_UVB = np.ma.masked_array(fits.open(path_SolarType_UVB)[0].data[1_000:], lamba_solar_UVB.mask)
Js_tell_VIS = np.ma.masked_array(fits.open(path_SolarType_VIS)[0].data[1_000:], lamba_solar_VIS.mask)
Js_tell_NIR = np.ma.masked_array(fits.open(path_SolarType_NIR)[0].data[1_000:], lamba_solar_NIR.mask)
Js_list = [Js_tell_UVB, Js_tell_VIS, Js_tell_NIR]
Js_tell_all = np.ma.concatenate((Js_tell_UVB, Js_tell_VIS, Js_tell_NIR))
Nub_median_west_all = np.concatenate((Nub_median_west[0], Nub_median_west[1], Nub_median_west[2]))
Nub_median_east_all = np.concatenate((Nub_median_east[0], Nub_median_east[1], Nub_median_east[2]))

#Fec_median_all = np.concatenate((Fec_median[0], Fec_median[1], Fec_median[2]))

# Global zones
regions = ['West Mare Imbrium', 'East Mare Imbrium', 'West Mare Nubium', 'East Mare Nubium',
           'Mare Fecundidatis']
flux_regions = [Imb_median_west, Imb_median_east, Nub_median_west, Nub_median_east, Fec_median]
lamba_regions = [lambaImb_median_west, lambaImb_median_east, lambaNub_median_west, lambaNub_median_east,
                 lambaFec_median]
color_regions = ['cyan', 'b', 'orange', 'y', 'm']
alpha_0 = 100
rv_star = -28.3
npoly_per_region_per_arm = [[4, 6, 0], [4, 6, 0], [5, 6, 6], [5, 6, 6], [4, 6, 4]]
rangeToNorm = [745, 755]
zoomx = None#[1000, 1025]
select_rv_UVB = [[380, 420], [430, 460], [480, 540]]
select_rv_VIS = [[640, 680], [698, 730], [780, 850], [870, 920]]
select_rv_NIR = [[1090, 1100], [1180, 1230], [1280, 1320], [1500, 1700], [2040, 2230]]#, [2325, 2330]]
select_rv = [select_rv_UVB, select_rv_VIS, select_rv_NIR]
fig1, axes1 = plt.subplots(3, 1, figsize=(20, 9), constrained_layout=True)
#fig2, axes2 = plt.subplots(3, 1, figsize=(20, 9))
for region, Ek_region, lamba_region, n_per_region, c in zip(regions, flux_regions, lamba_regions,
                                                            npoly_per_region_per_arm, color_regions):
    for idx, (band, n_per_band, select_rv_i) in enumerate(zip(['UVB', 'VIS', 'NIR'], n_per_region, select_rv)):
        if band == 'NIR' and 'Imbrium' in region:
            continue
        slit_width0 = 0.4 if band == 'UVB' else 0.5
        normXX_moon = normalizeXXnm(Ek_region[1], lamba_region[1], rangeToNorm)
        Alb_geom_zoneXX = geometricAlbedo(Ek_region[1], Js_list[1], alpha_0, slit_width0)
        normXX_alb = normalizeXXnm(Alb_geom_zoneXX, lamba_region[1], rangeToNorm)
        if region == 'East Mare Nubium':
            plot_zone_star_alb(fig1, axes1, lamba_region[idx], Ek_region[idx], lamba_solar_list[idx], Js_list[idx],
                               n_per_band, alpha_0, slit_width0, region, band, select=select_rv_i, rvmax=0, plot=False,
                               title=region + '-Hip098197-Albedo', color='k', zoomx=zoomx, lw=1., alpha_c=1., ls='-.')
        #plot_zone_star_alb(fig2, axes2, Ek_region[idx], Js_list[idx], alpha_0, slit_width0, lamba_region[idx],
        #                   lamba_solar_list[idx], region, band, title='Moon-Hip098197-Albedo', color=c, zoomx=None,
        #                   norm_moon=normXX_moon, norm_alb=normXX_alb
        #                   )
plt.show()

# BrokenXaxis
selectUVB = tuple([(329.8, 331.2), (331.8, 332.3), (332.8, 333.3), (333.4, 333.6), (334.8, 335.3)])
selectVIS = tuple([(588, 588.5), (588.8, 589.2), (589.2, 589.7), (880, 881), (1004.2, 1005.2)])
selectNIR = tuple([(2000, 2010), (2000, 2010), (2000, 2010), (2000, 2010), (2000, 2010), (2000, 2010)])

#norm_uvb_nub = fit_cont_v2(Nub_median_east[0], lambaNub_median_east[0], 4, 1, plot_cont=False)
#norm_solar_uvb_nub = fit_cont_v2(Js_list[0], lamba_solar_list[0], 4, 1, plot_cont=False)

plot_brokenxaxis(1, 1, [selectVIS], [[lambaNub_median_east[1], lamba_solar_list[1]]],
                 [[Nub_median_east[1], Js_list[1]]], 6, 100, 0.5, select=select_rv_VIS)





#norm_vis_nub = normalizeXXnm(Nub_median_west[1], lambaNub_median_west[1], rangeToNorm)
#norm_vis_solar = normalizeXXnm(Js_list[1], lamba_solar_list[1], rangeToNorm)

#vis_nub_norm = fit_cont_v2(Nub_median_east[2], lambaNub_median_east[2], 6, 1, plot_cont=False)
#vis_solar_norm = fit_cont_v2(Js_list[2], lamba_solar_list[2], 6, 1, plot_cont=False)

#lamba_solar_list[1].data[9_450:9_550]
#compute_dRV(lamba_solar_list[2].data[4230:4330], vis_solar_norm.data[4230:4330],
#            lambaNub_median_west[2].data[4230:4330], vis_nub_norm.data[4230:4330],
#            -50., 0., 0.1, skipedge=20)

#shift_pixels(lambaNub_median_west[0], 0.02, -44)

#plt.plot(lambaNub_median_west[1].data[4230:4330], vis_nub_norm.data[4230:4330], 'b.-')
#plt.plot(lamba_solar_list[1].data[4230:4330], vis_solar_norm.data[4230:4330], 'r.-')
#plt.plot(lamba_solar_list[1].data[4230:4330][3:], vis_solar_norm.data[4230:4330][:-3])
#spl_test = UnivariateSpline(lamba_solar_list[1].data/(1 - 28.3/C.c.to(u.km/u.s).value),
#                            lambaNub_median_east[1].data, s=0)
#plt.plot(lambaNub_median_east[1][7500:7600], vis_nub_norm[7500:7600], 'b.-')
#plt.plot(lamba_solar_list[1][7500:7600]/(1 - 28/C.c.to(u.km/u.s).value),
         #spl_test(lamba_solar_list[1][7500:7600]/(1 - 28/C.c.to(u.km/u.s).value)), 'r.-', lw=0.5)




#plt.plot(x_test[6500:7200], y_test[6500:7200], 'r.-')
#plt.grid()
#plt.show()
