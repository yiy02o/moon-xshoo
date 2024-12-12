from itertools import chain
import pandas
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from moon_after_molecfit import *
from scipy.optimize import curve_fit, minimize
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import convolve1d
from PyAstronomy import pyasl
import astropy.units as u
import astropy.constants as C
from brokenaxes import brokenaxes


def lambda_masked(nd_arr, condition):
    mask = mask_tellSpikes(nd_arr, condition)
    masked_data = np.ma.masked_array(nd_arr, mask)
    return masked_data


def geometricAlbedo(E_nu, E_s_nu, slit_width, E_nu_err, E_s_err, d_S_B=0.9938*u.AU, solar_source='sun',
                    d_E_B=399_184.62*u.km, show=True):
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
        # data ---> mejor separa las cosas correctas para que te quede de la forma
        # #A(lambda, alpha, i, e) = E_lunar(lambda, alpha, i, e)/Omega_lunar / E_sun(lambda)/pi
        E_ratio = E_nu.data / E_s_nu.data
        output_data = E_ratio * f_d * np.pi / Omega  # / D_ak
        output_data = np.ma.masked_array(output_data.value, mask=E_nu.mask)
        # error
        err_ratio_right = np.sqrt((E_nu_err.data/E_nu.data)**2 + (E_s_err.data/E_s_nu.data)**2)
        err_ratio = np.fabs(E_nu_err.data/E_s_err.data) * err_ratio_right
        output_err = err_ratio * f_d * np.pi / Omega  # / D_ak
        output_err = np.ma.masked_array(output_err.value, mask=E_nu_err.mask)
        if show:
            fig_err, ax_err = plt.subplots()
            ax_err.plot(output_err)
            #ax_err.plot(E_s_err.data, label="Solar irradiance model")
            ax_err.legend()
            ax_err.grid()
            ax_err.set_yscale("log")
            fig_err.show()
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
        if show:
            fig_err, ax_err = plt.subplots()
            ax_err.plot(output_err)
            ax_err.legend()
            ax_err.grid()
            ax_err.set_yscale("log")
            fig_err.show()
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


def fit_cont_v2(Ek, lamba, n_poly, s, n_poly_init=4, plot_cont=False, return_poly=False):
    lamba_clipped = lambda_ranges_v2(Ek, lamba, n_poly_init, s, plots_fit_cont=plot_cont)
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
        ax_fitMedian[0].set_title("Median region continuom fit and the polynomial")
        ax_fitMedian[0].plot(lamba.data, Ek.data, lw=0.5)
        ax_fitMedian[0].plot(lamba.data, poly_k, lw=0.5, c='r')
        # ax_fitMedian[0].set_yscale('log')
        ax_fitMedian[0].set_xlabel(r'$\lambda$ (nm)', fontsize=14)
        # plt.show()

        ax_fitMedian[1].set_title("Median region continuom fit normalized")
        ax_fitMedian[1].plot(lamba, Ek_norm, lw=0.5)
        # ax_fitMedian[1].set_yscale('log')
        ax_fitMedian[1].set_ylabel('Normalized flux', fontsize=14)
        ax_fitMedian[1].set_xlabel(r'$\lambda$ (nm)', fontsize=14)
        fig_fitMedian.show()
    if return_poly:
        return poly_k
    return Ek_norm


def bb_fitting(lims_edges_arr, lamba_arr, f_arr, p0, n_poly, plot_correct_bb=False):
    a0, temp0 = p0
    lims_uvb_vis, lims_vis_nir = lims_edges_arr
    mask_uvb_edge = np.logical_and(lamba_arr[0].data > lims_uvb_vis[0], lamba_arr[0].data < lims_uvb_vis[1])
    mask_vis_edge1 = np.logical_and(lamba_arr[1].data > lims_uvb_vis[0], lamba_arr[1].data < lims_uvb_vis[1])
    mask_vis_edge2 = np.logical_and(lamba_arr[1].data > lims_vis_nir[0], lamba_arr[1].data < lims_vis_nir[1])
    # we collapse into a common global mask
    mask_vis_edges = np.logical_or(mask_vis_edge1, mask_vis_edge2)
    mask_uvb_arm = np.logical_or(mask_uvb_edge, np.zeros(len(lamba_arr[0].data), dtype=bool))
    mask_vis_arm = np.logical_or(mask_vis_edges, lamba_arr[1].mask)
    mask_uvb_vis = np.concatenate((mask_uvb_arm, mask_vis_arm))
    lamba_uvb_vis = np.concatenate((lamba_arr[0].data, lamba_arr[1].data))
    f_uvb_vis = np.concatenate((f_arr[0].data, f_arr[1].data))
    lamba_to_fit = lamba_uvb_vis[~mask_uvb_vis]
    f_to_fit = f_uvb_vis[~mask_uvb_vis]
    mask_vis_only_edge2 = np.logical_or(~mask_vis_edge2, lamba_arr[1].mask)
    lamba_bad_edge = np.ma.masked_array(lamba_arr[1].data, mask=mask_vis_only_edge2)
    f_bad_edge = np.ma.masked_array(f_arr[1].data, mask=mask_vis_only_edge2)
    print(f_bad_edge)
    print(lamba_bad_edge)
    print(np.ma.polyfit(lamba_bad_edge, f_bad_edge, n_poly))
    poly_coeffs_bad_edge = np.ma.polyfit(lamba_bad_edge, f_bad_edge, n_poly)
    poly_function_bad_edge = np.poly1d(poly_coeffs_bad_edge)
    poly_bad_edge = poly_function_bad_edge(lamba_bad_edge.data)

    def BB(params, lamba_value):
        a, T_value = params
        temp = T_value*u.K
        lamba = lamba_value*u.nm
        output = a * 2*C.h*C.c**2 / lamba**5 / np.exp(C.h*C.c.to(u.nm/u.s)/(lamba*C.k_B*temp))
        return output.to(u.erg / u.cm**2 / u.s / u.AA).value

    def minimize_function(lamba_value, a, temp):
        params = a, temp
        return BB(params, lamba_value)

    p_opt, cov = curve_fit(minimize_function, lamba_to_fit, f_to_fit, p0=p0)
    poly_bb_uvb = BB(p_opt, lamba_arr[0].data)
    poly_bb_vis = BB(p_opt, lamba_arr[1].data)
    if plot_correct_bb:
        fig_curvefit, ax_curvefit = plt.subplots(figsize=(25, 5))
        ax_curvefit.plot(lamba_arr[0].data, poly_bb_uvb, c='k',
                         label=f"BB model with curve_fit: $A = {p_opt[0]}$ and $T = {p_opt[1]}$")
        ax_curvefit.plot(lamba_arr[1].data, poly_bb_vis, c='k')
        ax_curvefit.plot(lamba_to_fit, f_to_fit, c='b', alpha=0.6, lw=0.5, label='Data to fit')
        ax_curvefit.legend()
        ax_curvefit.grid()
        plt.show()
    f_vis_correction = f_arr[1].data * poly_bb_vis / poly_bad_edge
    f_vis_masked_data = np.where(mask_vis_edge2, f_vis_correction, f_arr[1].data)
    f_vis_masked = np.ma.masked_array(f_vis_masked_data, mask=f_arr[1].mask)
    return f_vis_masked, mask_vis_edge2


def poly_edges(lims_edges_arr, lamba_arr, f_arr, n_poly, method='yiyo_way', p0=None, plot_correct=False,
               which_edge='both'):
    """ Correct the missmatch on the edges of each arm of the whole wavelength range (UVB, VIS, NIR) by fcrrecting
    with a low order polynomial fit. lims_edges_arr is a two-list of two wavelengths limits in nm, which
    determines the zone to avoid when the polynomial fit is done (for ex.: [[500, 700], [800, 1500]]). lamba_arr and
    f_arr are the 3-list arms (2-list in case of Imbrium because of the lack of a NIR arm) masked arrays of wavelength
    and flux, respectively (for ex.: lambaNub_median_east and Nub_median_east).  """
    lims_uvb_vis = lims_edges_arr[0]
    lims_vis_nir = lims_edges_arr[1]
    list_corrected = []
    vis_edges_mask = []
    if plot_correct:
        fig_edges, ax_arms = plt.subplots(2, 1, figsize=(15, 5))
    fig_axes = fig_edges.axes if plot_correct else [1, 2]
    for ax_edges, edge in zip(fig_axes, ['uvb_vis', 'vis_nir']):
        if len(lamba_arr) == 2 and len(f_arr) == 2:
            if p0 is None:
                raise Exception("You have to give as an input a initial parameter to fit the BB")
            lamba_pair_arms = lamba_arr
            f_pair_arms = f_arr
            if edge == 'vis_nir':
                f_correction, mask_correction = bb_fitting(lims_edges_arr, lamba_arr, f_arr, n_poly=3, p0=p0, )
                list_corrected.append(f_correction)
                vis_edges_mask.append(mask_correction)
                continue
        elif len(lamba_arr) == 3 and len(f_arr) == 3:
            lamba_pair_arms = lamba_arr[:-1] if edge == 'uvb_vis' else lamba_arr[1:]
            f_pair_arms = f_arr[:-1] if edge == 'uvb_vis' else f_arr[1:]
        else:
            raise Exception('Check the whole spectrum dimensions that you have submitted')
        # for uvb_vis_edge
        lamba_data_arr = []       # lamba.data of the whole pair array
        f_data_arr = []           # flux.data of the whole pair array
        mask_arm_arr = []         # mask of the whole pair array, avoid the stron telluric absorption zones and edges
        only_mask_edge1 = []
        only_mask_edge2 = []
        mask_edges1_arr = []      # reverse mask to plot the uvb and vis edge
        mask_edges2_arr = []      # same as above but with the vis and nir edge
        mask_from_before_arr = []
        for lamba_arm, f_arm in zip(lamba_pair_arms, f_pair_arms):
            # we check and mask the masking edge criteria, for both edges uvb_vis and vis_nir
            mask_edge1 = np.logical_and(lamba_arm.data > lims_uvb_vis[0], lamba_arm.data < lims_uvb_vis[1])
            mask_edge2 = np.logical_and(lamba_arm.data > lims_vis_nir[0], lamba_arm.data < lims_vis_nir[1])
            # we collapse into a common global mask
            mask_edges = np.logical_or(mask_edge1, mask_edge2)
            # save the mask from telluric absorption
            mask_from_before = np.zeros(len(lamba_arm.data), dtype=bool) if lamba_arm.mask is False else lamba_arm.mask
            # we collapse into a common global mask (again)
            new_mask = np.logical_or(mask_edges, mask_from_before)
            # we collapse into a global reverse mask for the edges (both)
            new_mask_edges = np.logical_or(~mask_edges, mask_from_before)
            # lambda and flux masking arrays with only the edges (both)
            lamba_edges = np.ma.masked_array(lamba_arm.data, mask=new_mask_edges)
            f_edges = np.ma.masked_array(f_arm.data, mask=new_mask_edges)
            # lambda and flux arrays avoiding the edges and telluric absorption zones
            lamba_arm_masked = np.ma.masked_array(lamba_arm.data, mask=new_mask)
            f_arm_masked = np.ma.masked_array(f_arm.data, mask=new_mask)
            lamba_data_arr.append(lamba_arm.data)
            f_data_arr.append(f_arm.data)
            mask_arm_arr.append(new_mask)
            only_mask_edge1.append(mask_edge1)
            only_mask_edge2.append(mask_edge2)
            mask_edges1_arr.append(np.logical_or(~mask_edge1, mask_from_before))
            mask_edges2_arr.append(np.logical_or(~mask_edge2, mask_from_before))
            mask_from_before_arr.append(mask_from_before)
            if plot_correct:
                ax_edges.plot(lamba_arm_masked, f_arm_masked, c='grey', alpha=0.5, lw=0.5)
        if method == 'yiyo_way':
            # Collapse a pair of lambda and flux arrays into a single np.ma.masked_array() (avoiding edges and telluric)
            lamba_allwavs = np.ma.masked_array(list(chain.from_iterable(lamba_data_arr)),
                                               mask=list(chain.from_iterable(mask_arm_arr)))
            f_allwavs = np.ma.masked_array(list(chain.from_iterable(f_data_arr)),
                                           mask=list(chain.from_iterable(mask_arm_arr)))
            # Fit a polynomial on the whole pair
            poly_coeffs_allwavs = np.ma.polyfit(lamba_allwavs, f_allwavs, n_poly)
            poly_function = np.poly1d(poly_coeffs_allwavs)
            # Evaluate the polynomial in the whole pair lambda array, left lambda arm and right lambda arm
            poly_allwavs = poly_function(lamba_allwavs.data)
            poly_left_arm = poly_function(lamba_data_arr[0])
            poly_right_arm = poly_function(lamba_data_arr[1])
            poly_left_edge = norm_slope(f_pair_arms[0], lamba_pair_arms[0], 6)
            poly_right_edge = norm_slope(f_pair_arms[1], lamba_pair_arms[1], 6)
            left_correction = f_data_arr[0]*poly_left_arm/poly_left_edge
            f_left_corrected_data = np.where(only_mask_edge1[0] if edge == 'uvb_vis' else only_mask_edge2[0],
                                             left_correction, f_data_arr[0])
            f_left_corrected = np.ma.masked_array(f_left_corrected_data, mask=mask_from_before_arr[0])
            right_correction = f_data_arr[1]*poly_right_arm/poly_right_edge
            f_right_corrected_data = np.where(only_mask_edge1[1] if edge == 'uvb_vis' else only_mask_edge2[1],
                                              right_correction, f_data_arr[1])
            f_right_corrected = np.ma.masked_array(f_right_corrected_data, mask=mask_from_before_arr[1])
            if plot_correct:
                ax_edges.plot(lamba_allwavs, poly_allwavs, c='r',
                              label=f"Polynomial of order {n_poly} between the edges")
                ax_edges.plot(lamba_pair_arms[0], poly_left_edge, c='b', ls='--',
                              label=f"Polynomial of order {n_poly} of the left arm")
                ax_edges.plot(lamba_pair_arms[1], poly_right_edge, c='b', ls='--',
                              label=f"Polynomial of order {n_poly} of the right arm")
                ax_edges.plot(lamba_pair_arms[0], f_left_corrected, lw=0.5, c='k', label='Interface corrected')
                ax_edges.plot(lamba_pair_arms[1], f_right_corrected, lw=0.5, c='k')
            list_corrected.append([f_left_corrected, f_right_corrected])
            vis_edges_mask.append(only_mask_edge1[1] if edge == 'uvb_vis' else only_mask_edge2[0])
        elif method == 'pato_way':
            if type(n_poly) == int:
                raise Exception('You should give as an input a list of n_poly (two-list)')
            lamba_good_edge = np.ma.masked_array(lamba_data_arr[0], mask=mask_edges1_arr[0]) if edge == 'uvb_vis' \
                else np.ma.masked_array(lamba_data_arr[1], mask=mask_edges2_arr[1])
            f_good_edge = np.ma.masked_array(f_data_arr[0], mask=mask_edges1_arr[0]) if edge == 'uvb_vis' \
                else np.ma.masked_array(f_data_arr[1], mask=mask_edges2_arr[1])
            lamba_bad_edge = np.ma.masked_array(lamba_data_arr[1], mask=mask_edges1_arr[1]) if edge == 'uvb_vis' \
                else np.ma.masked_array(lamba_data_arr[0], mask=mask_edges2_arr[0])
            f_bad_edge = np.ma.masked_array(f_data_arr[1], mask=mask_edges1_arr[1]) if edge == 'uvb_vis' \
                else np.ma.masked_array(f_data_arr[0], mask=mask_edges2_arr[0])
            # good edge
            lamba_good_edge_clip = lambda_ranges_v2(f_good_edge, lamba_good_edge, 4, 1, plots_fit_cont=False)
            m_good_edge_clip = lamba_good_edge_clip.mask
            f_good_edge_clipped = np.ma.masked_array(f_good_edge.data, mask=m_good_edge_clip)
            poly_func_good_edge = np.poly1d(np.ma.polyfit(lamba_good_edge_clip, f_good_edge_clipped,
                                                          n_poly[0][0] if edge == 'uvb_vis' else n_poly[1][1]))
            poly_good_edge = poly_func_good_edge(lamba_good_edge_clip.data)
            # bad edge
            lamba_bad_edge_clip = lambda_ranges_v2(f_bad_edge, lamba_bad_edge, 4, 1, plots_fit_cont=False)
            m_bad_edge_clip = lamba_bad_edge_clip.mask
            f_bad_edge_clipped = np.ma.masked_array(f_bad_edge.data, mask=m_bad_edge_clip)
            poly_func_bad_edge = np.poly1d(np.ma.polyfit(lamba_bad_edge_clip, f_bad_edge_clipped,
                                                          n_poly[0][1] if edge == 'uvb_vis' else n_poly[1][0]))
            poly_bad_edge = poly_func_bad_edge(lamba_bad_edge_clip.data)
            # corrected edge
            poly_bad_edge_corrected = poly_func_good_edge(lamba_bad_edge.data)
            f_corrected_data = f_bad_edge*poly_bad_edge_corrected/poly_bad_edge
            correction = np.where(only_mask_edge1[1] if edge == 'uvb_vis' else only_mask_edge2[0], f_corrected_data,
                                  f_bad_edge)
            maskvis = mask_from_before_arr[1] if edge == 'uvb_vis' else mask_from_before_arr[0]
            f_corrected = np.ma.masked_array(correction, mask=maskvis)
            if plot_correct:
                ax_edges.plot(lamba_good_edge, f_good_edge, lw=0.5, c='b', alpha=0.5,
                              label='UVB edge' if edge == 'uvb_vis' else 'NIR edge')
                ax_edges.plot(lamba_bad_edge, f_bad_edge, lw=0.5, label='VIS edge', c='orange', alpha=0.5)
                ax_edges.plot(lamba_good_edge, poly_good_edge, c='b', ls='--',
                              label=f"Polynomial order {n_poly[0] if edge == 'uvb_vis' else n_poly[1]} "
                                    f"of the good edge")
                ax_edges.plot(lamba_bad_edge, poly_bad_edge, c='orange', ls='--',
                              label=f"Polynomial order {n_poly[0] if edge == 'uvb_vis' else n_poly[1]} of the bad edge")
                ax_edges.plot(lamba_bad_edge, f_corrected, lw=0.5, c='k', label='Corrected bad edge')
            list_corrected.append(f_corrected)
            print(1)
            vis_edges_mask.append(only_mask_edge1[1] if edge == 'uvb_vis' else only_mask_edge2[0])
        if plot_correct:
            ax_edges.legend(loc='best', fontsize='xx-small')
            ax_edges.grid()
            ax_edges.set_xlim(lims_uvb_vis[0] - 0.1 * (lims_uvb_vis[1] - lims_uvb_vis[0]),
                              lims_uvb_vis[1] + 0.1 * (lims_uvb_vis[1] - lims_uvb_vis[0])) \
                if edge == 'uvb_vis' else ax_edges.set_xlim(lims_vis_nir[0] - 0.1 * (lims_vis_nir[1] - lims_vis_nir[0]),
                                                            lims_vis_nir[1] + 0.1 * (lims_vis_nir[1] - lims_vis_nir[0]))
            ax_edges.set_ylim(np.ma.min(f_good_edge) if edge == 'vis_nir' else np.ma.min(f_bad_edge),
                              np.ma.max(f_bad_edge) if edge == 'vis_nir' else np.ma.max(f_good_edge))
    if method == 'yiyo_way':
        uvb_part = list_corrected[0][0]
        vis_part_edge1 = list_corrected[0][1]
        vis_part_edge2 = list_corrected[1][0] if len(lamba_arr) == 3 else list_corrected[1]
        vis_half_corrected = np.where(vis_edges_mask[0], vis_part_edge1.data, f_arr[1].data)
        vis_corrected = np.where(vis_edges_mask[1], vis_part_edge2.data, vis_half_corrected)
        vis_part = np.ma.masked_array(vis_corrected, mask=f_arr[1].mask)
        nir_part = list_corrected[1][1] if len(lamba_arr) == 3 else None
        final_list_corrected = [uvb_part, vis_part, nir_part] if len(lamba_arr) == 3 else [uvb_part, vis_part]
    if method == 'pato_way':
        if which_edge != 'uvb_vis' and which_edge != 'vis_nir' and which_edge != 'both':
            raise ValueError("Invalid edge correcting (try: uvb_vis, vis_nir or both)")
        vis_uvb_corrected = np.where(vis_edges_mask[0], list_corrected[0].data, f_arr[1].data)
        vis_nir_corrected = np.where(vis_edges_mask[1], list_corrected[1].data, f_arr[1].data)
        vis_all_corrected = np.where(vis_edges_mask[1], list_corrected[1].data, vis_uvb_corrected)
        vis_part = np.ma.masked_array(vis_uvb_corrected if which_edge == 'uvb_vis' else
                                      vis_nir_corrected if which_edge == 'vis_nir' else vis_all_corrected,
                                      mask=f_arr[1].mask)
        final_list_corrected = [f_arr[0], vis_part, f_arr[2]] if len(lamba_arr) == 3 else [f_arr[0], vis_part]
    if plot_correct:
        plt.show()
    return final_list_corrected


def broad_moon_spectra(tw, tf, res_down):
    maxsigma = 5
    smoothed_tf_spec, fwhm = pyasl.instrBroadGaussFast(tw.data, tf.data, res_down, edgeHandling='firstlast',
                                                       fullout=True, maxsig=maxsigma)
    interp_func = interp1d(tw.data, smoothed_tf_spec, kind='slinear')
    spectra_smoothed = interp_func(tw.data)
    spectra_smoothed = np.ma.masked_array(spectra_smoothed, mask=tf.mask)
    return spectra_smoothed


def rv_and_broadening_iteration(tw, lamba_solar, tf_norm, f_solar_norm_preBroad, f_solar, f_solar_err, select,
                                param_guesses, rvmin, rvmax, drv, skipedge, best_solars, dlamba, rv=None,
                                initial_resolution=None, plot=False, ax_lims=None, best_method='mean',
                                propagate_convErr=False, corr_data=False):
    """I have to write about this"""
    if rv is None:
        if select is None:
            raise ValueError(f"You should provide a wv range")
        if isinstance(select[0], (list, tuple)):
            idx_masked_tw = [np.where((tw.data > s[0]) & (tw.data < s[-1])) for s in select]
            idx_masked_w = [np.where((lamba_solar > s[0]) & (lamba_solar < s[-1])) for s in select]
            f_norm_broad_test_arr = [pyasl.instrBroadGaussFast(lamba_solar[m], f_solar_norm_preBroad[m],
                                                               param_guesses[0], edgeHandling="firstlast",
                                                               fullout=True)[0] for m in idx_masked_w]
            rv_per_chunk = [compute_dRV(lamba_solar[m_w], f_norm_broad, tw.data[m_tw], tf_norm.data[m_tw], rvmin, rvmax,
                                        drv, skipedge=skipedge, plot=False)
                            for (m_tw, m_w, f_norm_broad) in zip(idx_masked_tw, idx_masked_w, f_norm_broad_test_arr)]
            delta_per_chunk = [1 + rv_i / C.c.to(u.km / u.s).value for rv_i in rv_per_chunk]
            rv_chunks = [np.median(tw.data[m_tw]) for m_tw in idx_masked_tw]
            # if plot:
            #     fig_rvChunk, ax_rvChunk = plt.subplots(1, 1)
            #     ax_rvChunk.plot(rv_chunks, delta_per_chunk, 'b.-')
            #     ax_rvChunk.grid()
            #     ax_rvChunk.set_xlabel(f"$\\lambda$")
            #     ax_rvChunk.set_ylabel(f"$1 + \\Delta$")
            #     fig_rvChunk.show()

            rv = np.mean(rv_per_chunk)
    solar_wav_rvCorr = lamba_solar / (1 + rv / C.c.to(u.km / u.s).value)
    ################################################## ABSOLUTE SOLAR FLUX SHIFTED ####################################
    spl = UnivariateSpline(solar_wav_rvCorr, f_solar, s=0)
    solar_rvCorr = spl(tw)
    solar_rvCorr_w_rvCorr = spl(solar_wav_rvCorr)
    ################################# ABSOLUTE SOLAR FLUX UNC. SHIFTED (?) ############################################
    spl_err = UnivariateSpline(solar_wav_rvCorr, f_solar_err, s=0)
    solar_err_rvCorr = spl_err(tw)
    solar_err_rvCorr_w_rvCorr = spl_err(solar_wav_rvCorr)
    ################################ NORMALIZED ABSOLUTE SOLAR FLUX SHIFTED ###########################################
    spl_norm = UnivariateSpline(solar_wav_rvCorr, f_solar_norm_preBroad, s=0)
    solar_norm_rvCorr_tw = spl_norm(tw)   # solar spectra corrected by doppler shift, evaluated in template wavelength
    solar_norm_rvCorr_w_rvCorr = spl_norm(solar_wav_rvCorr)   # // // // // // //, // // solar wavelength corrected
    if plot:
        fig = plt.figure(layout='constrained', figsize=(18, 6))
        fig.suptitle(f"Broadening process")
        subfigs = fig.subfigures(1, 3, wspace=0.05, width_ratios=[1, 1, 1])
        subfigs[0].suptitle(f"Initial guesses $R = {np.round(param_guesses[0])}$ and "
                            f"$\sigma = {np.round(param_guesses[1], 2)}$")
        axsLeft = subfigs[0].subplots(3, 1)
        for idx, ax in enumerate(axsLeft):
            ax1 = ax.twinx()
            ax1.set_ylim(0.95, 1.05)
            mask_tw = np.logical_and(tw.data > ax_lims[idx][0], tw.data < ax_lims[idx][1])
            mask_solar_wav = np.logical_and(solar_wav_rvCorr > ax_lims[idx][0], solar_wav_rvCorr < ax_lims[idx][1])
            ax.plot(tw.data[mask_tw], tf_norm.data[mask_tw], 'k.-')
            ax1.plot(tw.data[mask_tw], tf_norm.data[mask_tw]/solar_norm_rvCorr_tw[mask_tw], 'b.-', alpha=0.7)
            ax.plot(solar_wav_rvCorr[mask_solar_wav], solar_norm_rvCorr_w_rvCorr[mask_solar_wav], c='g')

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
            tw_premasked = tw.data[np.abs(tw.data - line) <= dlamba]
            tf_norm_premasked = tf_norm.data[np.abs(tw.data - line) <= dlamba]
            solar_wav_rvCorr_masked = solar_wav_rvCorr[np.abs(solar_wav_rvCorr - line) <= dlamba]
            solar_norm_rvCorr_w_rvCorr_masked = solar_norm_rvCorr_w_rvCorr[np.abs(solar_wav_rvCorr - line) <= dlamba]
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
        if corr_data:
            raise ValueError("Try tomorrow, we still working on this...")
        else:
            meanWvl = np.mean(tw.data)
            fwhm_kernel = 1. / float(best_resolution) * meanWvl
            width_kernel = fwhm_kernel / (2. * np.sqrt(2. * np.log(2.)))
            print(f"Kernel width from the flux convolution process: {width_kernel}")
            width_kernel_variance = width_kernel / np.sqrt(2)
            print(f"Kernel width from the variance convolution process: {width_kernel_variance}")
            solar_flux_variance = solar_err_rvCorr_w_rvCorr**2
            convolved_solarFlux_variance = pyasl.broadGaussFast(solar_wav_rvCorr, solar_flux_variance,
                                                                width_kernel_variance, edgeHandling='firstlast',
                                                                maxsig=best_sigma)
            print("pasé la convolución del error")
            convolved_solarFlux_err = np.sqrt(convolved_solarFlux_variance / (2*np.sqrt(2*np.pi)*width_kernel_variance))
            # convolved_solarFlux_err = np.sqrt(convolved_solarFlux_variance)
            interp_solarFlux_err = interp1d(solar_wav_rvCorr, convolved_solarFlux_err, kind='slinear',
                                            fill_value="extrapolate")
            smooth_solarFlux_err = interp_solarFlux_err(tw.data)

    ################################### NORMALIZED ABSOLUTE SOLAR FLUX CONVOLVED #####################################

    convolved_solarFlux_norm, fwhm2 = pyasl.instrBroadGaussFast(solar_wav_rvCorr, solar_norm_rvCorr_w_rvCorr,
                                                                best_resolution, edgeHandling='firstlast', fullout=True,
                                                                maxsig=best_sigma)
    interp_func_norm = interp1d(solar_wav_rvCorr, convolved_solarFlux_norm, kind='slinear', fill_value="extrapolate")
    smooth_solarFlux_norm = interp_func_norm(tw.data)

    if plot:
        # Distribution of the parameters
        axsCenter = subfigs[1].subplots(3, 1)
        if initial_resolution is None:
            axsCenter[0].hist(result_arr[:, 0])
            axsCenter[0].set_xlabel('Resolution')
            axsCenter[0].axvline(best_resolution, c="k")
            axsCenter[1].plot(best_solars, result_arr[:, 0], label=r'$R_{\lambda}$', c="b")
            axsCenter[1].set_xlabel(f"$\\lambda$")
            axsCenter[1].set_ylabel(f"Resolution")
            axsCenter[1].axhline(best_resolution, c="k", ls="--")
            ax_sigma = axsCenter[1].twinx()
            ax_sigma.plot(best_solars, result_arr[:, 1], label=r'$\sigma_{\lambda}$', c="m")
            ax_sigma.set_ylabel(f"$\\sigma$")
            ax_sigma.axhline(best_sigma, c="grey", ls="--")
            axsCenter[1].legend(fontsize=7)
            ax_sigma.legend(fontsize=7)
            axsCenter[2].hist(result_arr[:, 1])
            axsCenter[2].set_xlabel(f"$\\sigma$")
            axsCenter[2].axvline(best_sigma, c="k", ls="--")
        # example of the broadened spectra
        subfigs[2].suptitle(f"Best parameters $R = {np.round(best_resolution)}$ and "
                            f"$\\sigma = {np.round(best_sigma, 2)}$")
        axsRight = subfigs[2].subplots(3, 1)
        for idx, ax in enumerate(axsRight):
            ax1 = ax.twinx()
            ax1.set_ylim(0.95, 1.05)
            mask_tw = np.logical_and(tw.data > ax_lims[idx][0], tw.data < ax_lims[idx][1])
            mask_solar_wav = np.logical_and(solar_wav_rvCorr > ax_lims[idx][0], solar_wav_rvCorr < ax_lims[idx][1])
            ax.plot(tw.data[mask_tw], tf_norm.data[mask_tw], 'k.-')
            ax1.plot(tw.data[mask_tw], tf_norm.data[mask_tw]/smooth_solarFlux_norm[mask_tw], 'b.-', alpha=0.7)
            ax.plot(tw.data[mask_tw], smooth_solarFlux_norm[mask_tw], c='g')
        fig.show()
        if propagate_convErr:
            fig_br, ax_br = plt.subplots(3, 1, figsize=(15, 6))
            for idx, ax in enumerate(ax_br):
                m_solar_err = np.logical_and(solar_wav_rvCorr > ax_lims[idx][0], solar_wav_rvCorr < ax_lims[idx][1])
                m_solar_br_err = np.logical_and(tw.data > ax_lims[idx][0], tw.data < ax_lims[idx][1])
                ax.plot(solar_wav_rvCorr[m_solar_err], f_solar_err[m_solar_err], 'r.-',
                        label="Unc. original: " + r'$\sqrt{Var(O_{x})}$')
                ax.plot(tw.data[m_solar_br_err], smooth_solarFlux_err[m_solar_br_err], 'k.-',
                        label="Unc. after convolution: " + r'$\sqrt{Var(C_{x})}$')
                ################## Secondary axis: u.erg / u.s / u.cm**2 / u.AA units #####################
                ax_toErg = ax.twinx()
                fsolar_err_toErg = (f_solar_err[m_solar_err] * u.W / u.m ** 2 / u.nm).to(u.erg / u.s / u.cm ** 2 / u.AA)
                smooth_solarFlux_err_toErg = (smooth_solarFlux_err[m_solar_br_err] * u.W / u.m ** 2 /
                                              u.nm).to(u.erg / u.s / u.cm ** 2 / u.AA)
                ax_toErg.plot(solar_wav_rvCorr[m_solar_err], fsolar_err_toErg.value, 'r.-',)
                ax_toErg.plot(tw.data[m_solar_br_err], smooth_solarFlux_err_toErg.value, 'k.-',)
                ax.grid()
                if idx == 1:
                    ax_toErg.set_ylabel(r'erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
            ax_br[1].legend()
            ax_br[1].set_ylabel(r'Error ($\sigma_{\lambda}$) ' + r'W $m^{-2}$ nm$^{-1}$')
            ax_br[2].set_xlabel(r'$\lambda$ (nm)')
            title_var = r'Uncorrelated data method, using ' \
                        r'Var$(C_{x}) = \sum_{z} K_{b}^{2}(x - z) \sigma_{z}^{2}$, ' \
                        f"where $b={width_kernel:.3f}$ " \
                        f"with $R \sim {(width_kernel*(2.*np.sqrt(2.*np.log(2.)))/meanWvl)**-1:.0f}$" if not corr_data \
                else "Not available"
            fig_br.suptitle(title_var)
            fig_br.show()
    return rv, [best_resolution, best_sigma], [smooth_solarFlux, smooth_solarFlux_err, smooth_solarFlux_norm]


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
    n_poly = 5 if in_uvb else 7 if in_vis else 7
    resolution_guess = 9_700 if in_uvb else 18_400 if in_vis else 11_600
    slit_width = 0.5 if in_uvb else 0.4
    ax_lims = [[406, 409], [476, 479], [501, 504]] if in_uvb else [[588, 591], [746, 749], [806, 809]] if in_vis \
        else [[1_240, 1_245], [1_590, 1_595], [2_100, 2_105]]
    tf_normalized = fit_cont_v2(tf, tw, n_poly, 1, plot_cont=False)
    f_solar_normalized = fit_cont_v2(flux_solar, lamba_solar, n_poly, 1, plot_cont=False)
    # First evaluation
    rv_0, best_params_0, smooth_solar_fluxes_0 = rv_and_broadening_iteration(tw, lamba_solar, tf_normalized,
                                                                             f_solar_normalized, flux_solar,
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

    ################### i have to improve the propagation of convolution error ##############################
    A_g, A_g_err = geometricAlbedo(tf, smooth_flux, slit_width, tf_err, smooth_flux_err, solar_source='sun', **kwargs)
    #print(f"Telluric zone in albedo: {A_g.data[A_g.mask][100:120]}")
    return rv_0, A_g, smooth_flux, A_g_err, smooth_flux_err


def oneForAll(tw, tf, tf_err, w=None, f=None, solar_source='sun', edge_corr_tf=None, edge_corr_f=None, rv=None,
              n_sigma=1, select=None, best_solar=None, drv=0.1, rvmin=-50, rvmax=50, skipedge=20, dlamba=10, plot=False,
              best_method='mean', initial_resolution=None, propagate_convErr=False, corr_data=False, **kwargs):
    """I have to re-write this """

    if solar_source not in ['sun', 'solar_twin', 'earthshine', 'comparing_moon_zones']:
        raise ValueError("solar source keyword invalid")
    # In the case of a non sun correction, check if w and f have been submitted
    if solar_source != 'sun' and np.logical_or(w is None, f is None):
        raise ValueError("You should introduce lambda (w) and flux (f) quantities as inputs")
    if isinstance(edge_corr_tf, list):
        # Check if there is any correction on the visible edges to perform
        lims = [[540, 590], [950, 1100]]
        tf = poly_edges(lims, edge_corr_tf[0], edge_corr_tf[1], [[2, 2], [3, 1]], p0=[5e-17, 4700], method='pato_way',
                        plot_correct=False, which_edge='vis_nir')[1]
        if solar_source != 'sun' and isinstance(edge_corr_f, list):  # f VIS arm is also corrected
            f = poly_edges(lims, edge_corr_f[0], edge_corr_f[1], [[3, 3], [2, 2]],
                           p0=[5e-17, 4700], method='pato_way', plot_correct=False,
                           which_edge='uvb_vis' if solar_source == 'solar_twin' else 'both')[1]
    in_uvb = np.any(np.logical_and(tw.data > 422., tw.data < 430.))
    slit_width = 0.5 if in_uvb else 0.4
    if solar_source == 'sun':
        return solar_smoothness(tw, tf, select, best_solar, tf_err, dlamba=dlamba, drv=drv, rvmin=rvmin, rvmax=rvmax,
                                skipedge=skipedge, plot=plot, best_method=best_method,
                                initial_resolution=initial_resolution, propagate_convErr=propagate_convErr,
                                corr_data=corr_data, **kwargs)
    ################################### I have to check for solar-star the uncertainities ###########################
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
    return geometricAlbedo(tf, f_interp, slit_width, solar_source='solar_twin', **kwargs), rv, \
        [tf_normalized, f_normalized_interp], f_interp,


def plot_global_reflectance(lamba_arr, data_arr, select_rv_arm_arr, data_err_arr, lamba1_arr=None, data1_arr=None,
                            solar_source='sun', best_solar_arm_arr=None, dlamba=20, x_lims=None, more_offsets=False,
                            plot_corrections=False, best_method='mean', initial_resolution=None, edge_correct_tf=None,
                            edge_correct_f=None, propagate_convErr=False, corr_data=False, **kwargs):
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
                                          edge_corr_tf=edge_correct_tf, edge_corr_f=edge_correct_f,
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
    print('Salí del print de OneForAll')
    if solar_source == 'solar_twin':
        for idx, (lamba_arm, data_arm, lamba1_arm, data1_arm, select_rv_arm, data_err_arm) in \
                enumerate(zip(lamba_arr, data_arr, lamba1_arr, data1_arr, select_rv_arm_arr, data_err_arr)):
            A_g_arm, rv_arm, Y_norm_arm, f_int, A_g_err_arm = oneForAll(lamba_arm, data_arm, data_err_arm, w=lamba1_arm,
                                                                        f=data1_arm, solar_source=solar_source,
                                                                        edge_corr_tf=edge_correct_tf,
                                                                        edge_corr_f=edge_correct_f,
                                                                        select=select_rv_arm, plot=plot_corrections,
                                                                        **kwargs)
            mask_arm = np.zeros(len(lamba_arm.data), dtype=bool) if isinstance(lamba_arm.mask, np.bool_) else \
                lamba_arm.mask
            A_g_data.append(A_g_arm.data)
            A_g_err.append(A_g_err_arm.data)
            rv.append(rv_arm)
            lamba_data.append(lamba_arm.data)
            lamba_mask.append(mask_arm)
            f_interp_data.append(f_int)
    lamba = np.ma.masked_array(list(chain.from_iterable(lamba_data)), mask=list(chain.from_iterable(lamba_mask)))
    A_g = np.ma.masked_array(list(chain.from_iterable(A_g_data)), mask=list(chain.from_iterable(lamba_mask)))
    A_g_err = np.ma.masked_array(list(chain.from_iterable(A_g_err)), mask=list(chain.from_iterable(lamba_mask)))
    #f_interp = np.ma.masked_array(list(chain.from_iterable(f_interp_data)), mask=list(chain.from_iterable(lamba_mask)))
    #f_interp_err = np.ma.masked_array(list(chain.from_iterable(f_interp_data_err)),
                                      #mask=list(chain.from_iterable(lamba_mask)))
    if x_lims is not None:
        mask_now = np.logical_or(lamba.mask, ~np.logical_and(lamba.data > x_lims[0], lamba.data < x_lims[1]))
        lamba = np.ma.masked_array(lamba.data, mask=mask_now)
        A_g = np.ma.masked_array(A_g.data, mask=mask_now)
        A_g_err = np.ma.masked_array(A_g_err.data, mask=mask_now)
    if more_offsets:
        # print(f"Telluric zone in convolved solar spectra: {f_interp_data[0].data[f_interp_data[0].mask][100:120]}")
        # print(f"Telluric zone in albedo: {A_g.data[A_g.mask][100:120]}")
        # print(f"Telluric zone wv range: {lamba.data[lamba.mask][100:120]}")
        return lamba, A_g, f_interp_data[0], A_g_err, f_interp_data_err[0]
    else:
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
#Nub_median_west_all = np.concatenate((Nub_median_west[0], Nub_median_west[1], Nub_median_west[2]))
#Nub_median_east_all = np.concatenate((Nub_median_east[0], Nub_median_east[1], Nub_median_east[2]))

#Fec_median_all = np.concatenate((Fec_median[0], Fec_median[1], Fec_median[2]))

# Global zones
regions = ['West Mare Imbrium', 'East Mare Imbrium', 'West Mare Nubium', 'East Mare Nubium',
           'Mare Fecundidatis']
#flux_regions = [Imb_median_west, Imb_median_east, Nub_median_west, Nub_median_east, Fec_median]
#lamba_regions = [lambaImb_median_west, lambaImb_median_east, lambaNub_median_west, lambaNub_median_east,
#                lambaFec_median]
color_regions = ['cyan', 'b', 'orange', 'y', 'm']
alpha_0 = 100
rv_star = -28.3
npoly_per_region_per_arm = [[4, 6, 0], [4, 6, 0], [5, 6, 6], [5, 6, 6], [4, 6, 4]]
rangeToNorm = [745, 755]
zoomx = [550, 560]
select_rv_UVB = [[390, 425], [430, 460], [480, 520], [525, 540]]
select_rv_VIS = [[580, 600], [610, 620], [638, 660], [730, 750], [790, 812], [825, 890]]
select_rv_NIR = [[1090, 1100], [1180, 1230], [1500, 1700]]#[1280, 1320], [1500, 1700]]
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
hl1 = [-26.971, 37.242]
hl2 = [-29.052, 37.404]
hl3 = [-30.850, 37.507]
hl4 = [-32.686, 37.616]
hl5 = [-34.637, 37.745]
hl6 = [-37.112, 37.969]
hl7 = [-39.408, 38.154]
hl8 = [-41.681, 38.325]
hl9 = [-44.046, 38.508]
hl10 = [-47.065, 38.795]
hl11 = [-49.851, 39.030]
hl12 = [-52.787, 39.272]
hl13 = [-56.072, 39.552]
hl14 = [-60.646, 40.000]
hl15 = [-65.464, 40.433]
hl16 = [-72.260, 41.041]
hl17 = [-82.803, 41.638]
hl18 = [-82.980, 40.591]
hl_PHcoo = [hl1, hl2, hl3, hl4, hl5, hl6, hl7, hl8, hl9, hl10, hl11, hl12, hl13, hl14, hl15, hl16, hl17, hl18]

################################### Mare Nubium + others (Mare in the fits) ###########################################
mare1 = [345.241, -26.600]
mare2 = [343.375, -26.423]
mare3 = [341.702, -26.289]
mare4 = [340.017, -26.151]
mare5 = [338.224, -25.986]
mare6 = [336.182, -25.756]
mare7 = [334.609, -25.648]
mare8 = [332.875, -25.496]
mare9 = [330.951, -25.296]
mare10 = [328.800, -25.040]
mare11 = [326.925, -24.860]
mare12 = [325.037, -24.678]
mare13 = [323.193, -24.511]
mare14 = [320.797, -24.205]
mare15 = [318.563, -23.949]
mare16 = [316.282, -23.691]
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
i_e_hl10 = [61, 64]
i_e_hl11 = [59, 66]
i_e_hl12 = [57, 68]
i_e_hl13 = [55, 70]
i_e_hl14 = [53, 73]
i_e_hl15 = [50, 77]
i_e_hl16 = [48, 82]
i_e_hl17 = [45, 90]
i_e_hl18 = [43, 90]
i_e_hl = [i_e_hl1, i_e_hl2, i_e_hl3, i_e_hl4, i_e_hl5, i_e_hl6, i_e_hl7, i_e_hl8, i_e_hl9, i_e_hl10, i_e_hl11, i_e_hl12,
          i_e_hl13, i_e_hl14, i_e_hl15, i_e_hl16, i_e_hl17, i_e_hl18]

################################### Mare Nubium + others (Mare in the fits) ###########################################
i_e_mare1 = [83, 27]
i_e_mare2 = [81, 28]
i_e_mare3 = [80, 29]
i_e_mare4 = [78, 30]
i_e_mare5 = [76, 32]
i_e_mare6 = [75, 33]
i_e_mare7 = [73, 34]
i_e_mare8 = [72, 35]
i_e_mare9 = [70, 37]
i_e_mare10 = [68, 38]
i_e_mare11 = [66, 40]
i_e_mare12 = [64, 41]
i_e_mare13 = [63, 43]
i_e_mare14 = [61, 45]
i_e_mare15 = [59, 47]
i_e_mare16 = [56, 49]
i_e_mare = [i_e_mare1, i_e_mare2, i_e_mare3, i_e_mare4, i_e_mare5, i_e_mare6, i_e_mare7, i_e_mare8, i_e_mare9,
            i_e_mare10, i_e_mare11, i_e_mare12, i_e_mare13, i_e_mare14, i_e_mare15, i_e_mare16]
