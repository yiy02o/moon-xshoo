import numpy as np

from get_albedovalues import *
from disk_and_phaseFunctions import *
from scipy.spatial import ConvexHull
from velikodsky_spectrum import *


def median_filter(x, k):
    assert k % 2 == 1
    assert x.ndim == 1
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.ma.median(y, axis=1)


def fit_cont_v3(Ek, lamba, n_poly, s, n_poly_init=4, plot_cont=False, return_poly=False):
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
        fig_fitMedian, ax_fitMedian = plt.subplots(2, 1, layout="constrained")
        ax_fitMedian[0].set_title("Median region continuom fit and the polynomial")

        ax_fitMedian[0].plot(lamba, Ek, lw=0.5)
        ax_fitMedian[0].plot(lamba, poly_k, lw=0.5, c='r')
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


def flat_reflectance(wv, al, al_err, lims_to_fit, date_id, method="SecondFirst", newMask=None, show=False,
                     lims_shoulders=None, n_poly_init=1, s=1, showFitmethod=False):
    """Remove the continuum reflectance using two methods: ConvexHull and SecondFirst order polynomial"""
    if newMask is not None:
        wv = np.ma.masked_array(wv.data, mask=newMask)
        al = np.ma.masked_array(al.data, mask=newMask)
        al_err = np.ma.masked_array(al_err.data, mask=newMask)
    maskRegion_toFit = ~np.logical_and(wv.data > lims_to_fit[0], wv.data < lims_to_fit[-1])
    m_poly = np.logical_or(wv.mask, maskRegion_toFit)
    wv_to_flat = np.ma.masked_array(wv.data, mask=m_poly)
    alb_to_flat = np.ma.masked_array(al.data, mask=m_poly)
    alb_err_to_flat = np.ma.masked_array(al_err.data, mask=m_poly)
    if method == "SecondFirst":
        if lims_shoulders is None:
            raise ValueError("You should provide a wavelength range to fit the second order polynomial")
        # x_range1, x_range2 = [lims_to_fit[0], lims_to_fit[0]+3], [lims_to_fit[-1]-1, lims_to_fit[-1]]
        x_mask1, x_mask2 = np.logical_and(wv_to_flat.data > lims_shoulders[0][0],
                                          wv_to_flat.data < lims_shoulders[0][-1]), \
            np.logical_and(wv_to_flat.data > lims_shoulders[-1][0], wv_to_flat.data < lims_shoulders[-1][-1])
        x_arr = wv_to_flat.data[np.logical_or(x_mask1, x_mask2)]
        y_arr = alb_to_flat.data[np.logical_or(x_mask1, x_mask2)]
        # first fit
        poly_first = norm_slope(y_arr, x_arr, n_poly_init)
        y_norm_first = y_arr / poly_first
        y_clipped = sigma_clip(y_norm_first, sigma=s, stdfunc=scipy.stats.iqr)
        m_clip = y_clipped.mask
        lamba_clipped = np.ma.masked_array(x_arr, mask=m_clip)
        y_clipped = np.ma.masked_array(y_arr, mask=m_clip)
        # final fit
        coeff_1umBand = np.ma.polyfit(lamba_clipped, y_clipped, 2)
        polyFunction_1umBand = np.poly1d(coeff_1umBand)
        poly_1umBand = polyFunction_1umBand(wv_to_flat.data)
        flat_alb = np.ma.masked_array(al.data / poly_1umBand, mask=wv_to_flat.mask)
        flat_alb_err = np.ma.masked_array(al_err.data / poly_1umBand, mask=wv_to_flat.mask)
        if showFitmethod:
            fig_fit, ax_fit = plt.subplots(1, 1, sharex=True)
            ax_fit.plot(x_arr, y_arr, '.', c="b", label="Data to fit", ms=2)
            ax_fit.plot(lamba_clipped, y_clipped, '.', c="r", label="Data clipped", ms=2)
            ax_fit.plot(wv_to_flat, poly_1umBand, c="k", alpha=.5, label="Second-order polynomial")
            ax_fit.legend()
            ax_fit.grid()
            ax_fit.set_ylabel(r'$A(\lambda, \alpha, i, e)$')
            ax_fit.set_xlabel(r'$\lambda$ (nm)')
            fig_fit.show()

    else:
        pass
    if show:
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(wv_to_flat, alb_to_flat, c="b", lw=.5, label="Data")
        ax[0].plot(wv_to_flat, poly_1umBand, c="r", ls="--", label="Second order fit")
        ax[0].grid()
        ax[0].axvspan(lims_shoulders[0][0], lims_shoulders[0][-1], facecolor="g", alpha=.2)
        ax[0].axvspan(lims_shoulders[-1][0], lims_shoulders[-1][-1], facecolor="g", alpha=.2)
        ax[0].set_ylabel(r'$A(\lambda, \alpha, i, e)$')
        ax[1].plot(wv_to_flat, flat_alb, c="b", lw=.5)
        ax[1].axhline(1, ls="--", c="r")
        ax[1].grid()
        ax[1].axvspan(lims_shoulders[0][0], lims_shoulders[0][-1], facecolor="g", alpha=.2)
        ax[1].axvspan(lims_shoulders[-1][0], lims_shoulders[-1][-1], facecolor="g", alpha=.2)
        ax[1].set_ylabel(r'Continuum removed')
        ax[2].plot(wv_to_flat, flat_alb_err,)
        ax[2].grid()
        ax[2].set_ylabel(r'$\sigma_{\lambda}$')
        ax[2].set_xlabel(r'$\lambda$ (nm)')
        ax[0].legend(fontsize=8, loc="upper left")
        fig.suptitle(date_id)
        fig.show()
    return wv_to_flat, flat_alb, flat_alb_err, poly_1umBand


def mgm(params, lamba):
    s1, s2, s3, mu1, mu2, mu3, sigma1, sigma2, sigma3 = params
    output = 1 + s1 * np.exp(-(lamba**(-1) - mu1**(-1))**2 / (2 * sigma1**2)) + \
             s2 * np.exp(-(lamba**(-1) - mu2**(-1))**2 / (2 * sigma2**2)) + \
             s3 * np.exp(-(lamba**(-1) - mu3**(-1))**2 / (2 * sigma3**2))
    return output


def mgm_individual(params, lamba):
    s, mu, sigma = params
    output = 1 + s * np.exp(-(lamba ** (-1) - mu ** (-1)) ** 2 / (2 * sigma ** 2))
    return output


def minimize_function(lamba, s1, s2, s3, mu1, mu2, mu3, sigma1, sigma2, sigma3):
    params = s1, s2, s3, mu1, mu2, mu3, sigma1, sigma2, sigma3
    return mgm(params, lamba)


def check_lims(X, Y, lims, max_mode="max", min_mode="min", s_max=3, s_min=3, list_type=False):
    mask_lims_arr = [np.logical_and(x.data > lims[0], x.data < lims[1]) for x in X]
    new_masked_arr = [np.ma.masked_array(y.data, mask=np.logical_or(y.mask, ~m)) for y, m in zip(Y, mask_lims_arr)]
    y_new_arr = [y_m.data[~y_m.mask] for y_m in new_masked_arr]
    if max_mode == "max":
        max_list = [np.max(y[~np.isnan(y)]) for y in y_new_arr]
    elif max_mode == "median":
        max_list = [np.median(y[~np.isnan(y)]) + s_max*np.std(y[~np.isnan(y)]) for y in y_new_arr]
    if min_mode == "min":
        min_list = [np.min(y[~np.isnan(y)]) for y in y_new_arr]
    elif min_mode == "median":
        min_list = [np.median(y[~np.isnan(y)]) - s_min*np.std(y[~np.isnan(y)]) for y in y_new_arr]
    if list_type:
        return min_list, max_list
    else:
        return np.min(min_list), np.max(max_list)


def return_telluricData(tw, tf, uncorr=True):
    mask_tell = tw.mask
    if uncorr:
        mask_corr = np.logical_and(tw.data > 927, tw.data < 1_022.3)
        mask_corr_and_tell = np.logical_and(mask_corr, mask_tell)
        mask_tell = np.logical_and(mask_tell, ~mask_corr_and_tell)
    tf_tell = np.ma.masked_array(tf.data, mask=~mask_tell)
    tw_tell = np.ma.masked_array(tw.data, mask=~mask_tell)
    return tw_tell, tf_tell


def normalizeXXnm_v2(wv, al, norm_range):
    mask_range = ~np.logical_and(wv.data > norm_range[0], wv.data < norm_range[1])
    mask_from_before = al.mask
    new_mask = np.logical_or.reduce(np.asarray([mask_range, mask_from_before]))
    y_arr_masked = np.ma.masked_array(al.data, mask=new_mask)
    median_to_norm = np.ma.median(y_arr_masked)
    y_normXXnm = al.data / median_to_norm
    y_normXXnm = np.ma.masked_array(y_normXXnm, mask=al.mask)
    return y_normXXnm


def spectral_ratio(wv, al, al_ref, norm_range):
    al_normXX = normalizeXXnm_v2(wv, al, norm_range)
    al_ref_normXX = normalizeXXnm_v2(wv, al_ref, norm_range)
    al_ratio = al_normXX / al_ref_normXX
    return al_ratio


def radf30_0_30(wv_arr, radf_obs, alpha_deg, i_deg, e_deg, eta=.34, disk_func="akimov", phase_func="hicks",
                mare=True, method_interp="CS", bc_type="natural"):
    i_to_rad = i_deg * np.pi / 180
    e_to_rad = e_deg * np.pi / 180
    alpha_to_rad = alpha_deg * np.pi / 180

    ########################### Xl observed ###############################
    cos_phi = (np.cos(alpha_to_rad) - np.cos(i_to_rad) * np.cos(e_to_rad)) / (np.sin(i_to_rad) * np.sin(e_to_rad))
    phi = np.arccos(cos_phi)
    num_sqrt = np.sin(i_to_rad + e_to_rad)**2 - np.sin(2*i_to_rad) * np.sin(2*e_to_rad) * np.cos(phi/2)**2
    den_sqrt = np.sin(i_to_rad + e_to_rad)**2 - np.sin(2*i_to_rad) * np.sin(2*e_to_rad) * np.cos(phi/2)**2 + \
               (np.sin(e_to_rad) * np.sin(i_to_rad) * np.sin(alpha_to_rad))**2
    cos_beta = np.sqrt(num_sqrt / den_sqrt)
    la_to_rad = np.arccos(cos_beta)
    cos_gamma = np.cos(e_to_rad) / np.cos(la_to_rad)
    lo_to_rad = np.arccos(cos_gamma)

    ########################### Xl(30, 0, 30) ###############################
    alpha_30_0_30 = 30 * np.pi / 180
    i_30_0_30 = 30 * np.pi / 180
    e_30_0_30 = 0 * np.pi / 180
    la_30_0_30 = np.arccos(1)
    lo_30_0_30 = np.cos(e_30_0_30) / np.cos(la_30_0_30)

    if disk_func != "akimov" and disk_func != "LS" and disk_func != "mcEwen":
        raise ValueError("The disk function that you provided is not valid.")
    if disk_func == "akimov":
        Xl_obs = akimov_disk(lo_to_rad, la_to_rad, alpha_to_rad, eta)          # lo, la and alpha are in rad
        Xl_30_0_30 = akimov_disk(lo_30_0_30, la_30_0_30, alpha_30_0_30, eta)   # photometric coordinates are given
    if disk_func == "LS":
        Xl_obs = LS_disk(i_to_rad, e_to_rad)                                   # i and e are given in radians
        Xl_30_0_30 = LS_disk(i_30_0_30, e_30_0_30)
    if disk_func == "mcEwen":
        Xl_obs = mcEwen_disk(i_to_rad, e_to_rad, alpha_deg)                    # i and e are given in radians,
        Xl_30_0_30 = mcEwen_disk(i_30_0_30, e_30_0_30, 30)                     # however alpha is given in deg
        pass

    # f(g) and f(30)
    if phase_func != "hicks" and phase_func != "buratti" and phase_func != "hillier":
        raise ValueError("The phase function that you provided is not valid.")
    if phase_func == "hicks":
        # alpha is given in deg, 30 deg
        f_obs = hicks2011(wv_arr, alpha_deg, mare=mare, method_interp=method_interp, bc_type=bc_type)
        f_30 = hicks2011(wv_arr, 30, mare=mare, method_interp=method_interp, bc_type=bc_type)
    if phase_func == "buratti":
        # alpha is given in deg, 30 deg
        f_obs = buratti2011(wv_arr, alpha_deg, mare=mare, method_interp=method_interp, bc_type=bc_type)
        f_30 = buratti2011(wv_arr, 30, mare=mare, method_interp=method_interp, bc_type=bc_type)
    if phase_func == "hillier":
        # alpha is given in deg, 30 deg
        f_obs = hillier1999(wv_arr, alpha_deg, mare=True, method_interp=method_interp, bc_type=bc_type)
        f_30 = hillier1999(wv_arr, 30, mare=True, method_interp=method_interp, bc_type=bc_type)

    output = radf_obs * Xl_30_0_30 * f_30 / (Xl_obs * f_obs)
    return output


def eq_albedo(radf_obs, alpha, la, lo, eta=.34):
    """This function computes the equigonal albedo  defined as:
    A(lambda, alpha, i, e) = A_eq(lambda, alpha) * D(alpha, i, e), where A is the apparent albedo defined in
    moon_reflectance.py (radf_obs) and D is the disk function which determinate how the brightness its distributed
    across the lunar disk. We use the Akimov function in this case. The parameters i and e can be defined in function
    of a more comfortable variables called the photometric longitudes and latitudes (lo, la). eta is a coeff set to .34
    Akimov disk function can be found in eq. 3 of Velikodsky et al (2011)."""
    la_to_rad = la*np.pi/180
    if lo < 0:
        lo_to_rad = np.fabs(lo) * np.pi/180
    elif np.logical_and(360 > lo, lo > 180):
        lo_to_rad = (360 - lo) * np.pi / 180
    else:
        lo_to_rad = lo * np.pi / 180
    alpha_to_rad = alpha*np.pi/180
    left_term_diskAk = (np.cos(la_to_rad))**(eta*alpha_to_rad/(np.pi - alpha_to_rad))
    mid_term_diskAk = np.cos(alpha_to_rad/2) / np.cos(lo_to_rad)
    right_term_diskAk = np.cos((lo_to_rad - alpha_to_rad/2)*np.pi/(np.pi - alpha_to_rad))
    D_ak = left_term_diskAk * mid_term_diskAk * right_term_diskAk
    output = radf_obs / D_ak
    return output


def convexHull(points, show=False):
    x, y = points.T
    augmented = np.concatenate([points, [(x[0], np.min(y) - 1), (x[-1], np.min(y) - 1)]], axis=0)
    hull = ConvexHull(augmented)
    continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
    continuum_function = interp1d(*continuum_points.T)
    yprime = y / continuum_function(x)
    if show:
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(x, y, label="Data")
        axes[0].plot(*continuum_points.T, label="Continuum")
        axes[0].legend()
        axes[1].plot(x, yprime, label="Data / Continuum")
        axes[1].legend()
    return np.c_[x, yprime]


def fe_idx(wv, al, deltawv1=1, deltawv2=1, show=False):
    # compute the median ref. around 757 nm and 918 nm
    # 757 nm
    mask757nm = ~np.logical_and(wv.data > 757 - deltawv1, wv.data < 757 + deltawv1)
    m_median757nm = np.logical_or(wv.mask, mask757nm)
    wv757nm = np.ma.masked_array(wv.data, mask=m_median757nm)
    al757nm = np.ma.masked_array(al.data, mask=m_median757nm)
    R757nm = np.ma.median(al757nm)
    # 918 nm
    mask918nm = ~np.logical_and(wv.data > 918 - deltawv2, wv.data < 918 + deltawv2)
    m_median918nm = np.logical_or(wv.mask, mask918nm)
    wv918nm = np.ma.masked_array(wv.data, mask=m_median918nm)
    al918nm = np.ma.masked_array(al.data, mask=m_median918nm)
    R918nm = np.ma.median(al918nm)
    if show:
        fig_fe, ax_fe = plt.subplots(2, 1)
        ax_fe[0].plot(wv757nm.data[~wv757nm.mask], al757nm.data[~wv757nm.mask])
        ax_fe[0].grid()
        ax_fe[1].plot(wv918nm.data[~wv918nm.mask], al918nm.data[~wv918nm.mask])
        ax_fe[1].grid()
        ax_fe[1].set_xlabel(r'$\lambda$ (nm)')
        fig_fe.show()
    Fe_index = np.arctan(((R918nm / R757nm) - 1.19) / (R757nm - .06))
    wtFeO = 8.878 * Fe_index**1.8732
    return Fe_index, wtFeO, R757nm, R918nm / R757nm


WVTOFITMARE = []
ALBFLATMARE = []
ALBFLATERRMARE = []
POLYCONTMARE = []

fig_appAl, ax_appAl = plt.subplots()
for id_num, (wv_global, alb, alb_err, date, c) in enumerate(zip(WVGLOBAL_MARE[0:8], ALBGLOBAL_MARE[0:8],
                                                                ALBGLOBALERR_MARE[0:8], DATESGLOBAL_MARE[0:8],
                                                                ["m", "lime"])):
    wvtofit, albflat, alberrflat, polycont = flat_reflectance(wv_global, alb, alb_err, [590, 1_550],
                                                              date[0], method="SecondFirst", newMask=None, show=True,
                                                              lims_shoulders=[[600, 750], [1_485, 1_550]],
                                                              showFitmethod=True)
    ax_appAl.plot(wv_global, alb, label=f"Site: {id_num+1}", c=c)
    ax_appAl.plot(wv_global, median_filter(alb, 11), c="k", lw=.8)
    ax_appAl.plot(np.ma.masked_array(wv_global.data, mask=~wv_global.mask),
                  np.ma.masked_array(alb.data, mask=~alb.mask), c="grey", alpha=.3)
    WVTOFITMARE.append(wvtofit)
    ALBFLATMARE.append(albflat)
    ALBFLATERRMARE.append(alberrflat)
    POLYCONTMARE.append(polycont)
ax_appAl.set_ylim(np.ma.min(ALBGLOBAL_MARE[0:8]), np.ma.max(ALBGLOBAL_MARE[0:8]))
ax_appAl.grid()
ax_appAl.legend()
ax_appAl.set_xlabel(f"$\lambda$ (nm)")
ax_appAl.set_ylabel(r'$A(\lambda, \alpha, i, e)$')
fig_appAl.show()


# fig_pre_and_post, ax_pre_and_post = plt.subplots()
# for id_num, (wv_global, albflat, tf_post,
#              wv_globaltell, albtell,
#              tf_pre, polycont) in enumerate(zip(WVGLOBAL_MARE[:1], ALBFLATMARE[:1], TFGLOBAL_MARE[:1],
#                                                 WVGLOBAL_MARE_preMolecfit[:1], ALBGLOBAL_MARE_preMolecfit[:1],
#                                                 TFGLOBAL_MARE_preMolecfit[:1], POLYCONTMARE[:1])):
#     ax_pre_and_post.plot(wv_globaltell, albtell / polycont, label="Pre-telluric correction", c="b")
#     ax_pre_and_post.plot(wv_global, albflat,
#                          label="Post-telluric correction (grey shadows represent bad telluric corrections)", c="orange")
#     ax_pre_and_post.plot(wv_global, median_filter(albflat, 11), label="Post-telluric correction using median filter",
#                          c="k", lw=.8)
#     ax_pre_and_post.plot(np.ma.masked_array(wv_global.data, mask=~wv_global.mask),
#                          np.ma.masked_array(albflat.data, mask=~albflat.mask), c="grey", alpha=.3)
# ax_pre_and_post.set_xlim(600, 1_550)
# ax_pre_and_post.set_ylim(.7, 1.1)
# ax_pre_and_post.legend(fontsize=7)
# ax_pre_and_post.grid()
# ax_pre_and_post.set_xlabel(f"$\lambda$ (nm)")
# ax_pre_and_post.set_ylabel(r'Continuum removed')
# fig_pre_and_post.show()


# fig_phaseVelik, ax_phaseVelik = plt.subplots()


def cut_spectra(wv, spec, lims):
    mask_lims = ~np.logical_and(wv.data > lims[0], wv.data < lims[1])
    mask_tell = wv.mask
    mask_equivalent = np.logical_or(mask_lims, mask_tell)
    wv_masked = np.ma.masked_array(wv.data, mask=mask_equivalent)
    spec_masked = np.ma.masked_array(spec.data, mask=mask_equivalent)
    return wv_masked, spec_masked


# def velikPhaseFunction(wv, al, lims):
#     wv_data, al_data = cut_spectra(wv, al, lims)
#     # Velik lims
#     mask_lims = np.logical_and(velik_table[:, 0] > lims[0], velik_table[:, 0] < lims[1])
#     wv_velik, alEqNorm_velik = velik_table[:, 0][mask_lims], eqAl_recovered[mask_lims]
#     ph_velikFunc_interp = scipy.interpolate.CubicSpline(wv_velik, alEqNorm_velik, bc_type="natural")
#     ph_velik_data = ph_velikFunc_interp(wv_data)
#     c_norm =
#     pass



fig_30_0_30, ax_30_0_30 = plt.subplots(2, 3, figsize=(15, 6), layout="constrained", sharex=True)
ax_ROLO = ax_30_0_30[0, 0].twinx()
ax_M3 = ax_30_0_30[0, 1].twinx()
ax_Clementine = ax_30_0_30[0, 2].twinx()
ax_ratio_ROLO = ax_30_0_30[1, 0].twinx()
ax_ratio_M3 = ax_30_0_30[1, 1].twinx()
ax_ratio_Clementine = ax_30_0_30[1, 2].twinx()

ROLO_lims = [347, 2_390]
M3_lims = [460.99, 2_936.27]
Clementine_lims = [415, 1_000]


max_axtwinx = 0
min_axtwinx = 0
max_radf = 0
min_radf = 0
bbox_radf = dict(boxstyle="round", fc="0.9")
ALB30_0_30 = []
alpha_epoch = 86
for id_num, (wv_global, alb, alb_err, date, ph_coo) in enumerate(zip(WVGLOBAL_MARE[0:8], ALBGLOBAL_MARE[0:8],
                                                                      ALBGLOBALERR_MARE[0:8], DATESGLOBAL_MARE[0:8],
                                                                      i_e_mare[0:8])):
    in_i, e_i = ph_coo
    radf_cali_ROLO = radf30_0_30(wv_global, median_filter(alb, 11), alpha_epoch, in_i, e_i, eta=.34,
                                 disk_func="LS", phase_func="buratti", mare=True, method_interp="CS",
                                 bc_type="natural")
    radf_cali_M3 = radf30_0_30(wv_global, median_filter(alb, 11), alpha_epoch, in_i, e_i, eta=.34,
                               disk_func="LS", phase_func="hicks", mare=True, method_interp="CS",
                               bc_type="natural")
    radf_cali_Clementine = radf30_0_30(wv_global, median_filter(alb, 11), alpha_epoch, in_i, e_i, eta=.34,
                                       disk_func="LS", phase_func="hillier", mare=True, method_interp="CS",
                                       bc_type="natural")

    # ROLO
    wv_ROLO, radf_ROLO = cut_spectra(wv_global, radf_cali_ROLO, ROLO_lims)
    ax_30_0_30[1, 0].plot(wv_ROLO, radf_ROLO)
    ax_30_0_30[0, 0].plot(wv_global, median_filter(alb, 11))
    ph_buratti100 = buratti2011(wv_buratti, alpha_epoch, mare=True)
    ph_buratti30 = buratti2011(wv_buratti, 30, mare=True)
    buratti100_coeff = np.polyfit(wv_buratti, ph_buratti100, 2)
    
    if id_num == 0:
        ph_buratti100Interpolation = buratti2011(wv_ROLO, alpha_epoch, mare=True)
        ph_buratti30Interpolation = buratti2011(wv_ROLO, 30, mare=True)
        ax_ROLO.plot(wv_buratti, ph_buratti100, "x", c="grey")
        ax_ROLO.plot(wv_buratti, ph_buratti30, "v", c="k")
        ax_ROLO.plot(wv_ROLO.data, ph_buratti100Interpolation, ls="--", c="grey")
        ax_ROLO.plot(wv_ROLO.data, ph_buratti30Interpolation, ls="--", c="k")
        ax_ratio_ROLO.plot(wv_ROLO.data, ph_buratti30Interpolation/ph_buratti100Interpolation,
                           c="r", ls="--", alpha=.6)
        ax_ROLO.annotate(f"ROLO", (2_000, 0.05), bbox=bbox_radf)
        ax_ROLO.annotate(f"Buratti + LS", (1_910, 0.035))
        ax_ROLO.annotate(f"$\\alpha \\in (0, 90)°$", (1_920, 0.025))

    # M3
    wv_M3, radf_M3 = cut_spectra(wv_global, radf_cali_M3, M3_lims)
    ax_30_0_30[1, 1].plot(wv_M3, radf_M3, label=f"Site: {id_num + 1}")
    ax_30_0_30[0, 1].plot(wv_global, median_filter(alb, 11))
    ph_hicks100 = hicks2011(wv_hicks, alpha_epoch, mare=True)
    ph_hicks30 = hicks2011(wv_hicks, 30, mare=True)
    if id_num == 0:
        ph_hicks100Interpolation = hicks2011(wv_M3, alpha_epoch, mare=True)
        ph_hicks30Interpolation = hicks2011(wv_M3, 30, mare=True)
        ax_M3.plot(wv_hicks, ph_hicks100, "x", c="grey", label=f"$\\alpha={alpha_epoch}°$")
        ax_M3.plot(wv_hicks, ph_hicks30, "v", c="k", label=f"$\\alpha=30°$")
        ax_M3.plot(wv_M3.data, ph_hicks100Interpolation, ls="--", c="grey")
        ax_M3.plot(wv_M3.data, ph_hicks30Interpolation, ls="--", c="k")
        ax_ratio_M3.plot(wv_M3.data, ph_hicks30Interpolation / ph_hicks100Interpolation,
                         c="r", ls="--", alpha=.6, label=f"f(30°)/f({alpha_epoch}°)")
        ax_M3.annotate(f"M3", (2_000, 0.05), bbox=bbox_radf)
        ax_M3.annotate(f"Hicks + LS", (1_870, 0.035))
        ax_M3.annotate(f"$\\alpha \\in (0, 80)°$", (1_870, 0.025))

    # Clementine
    wv_Clementine, radf_Clementine = cut_spectra(wv_global, radf_cali_Clementine, Clementine_lims)
    ax_30_0_30[1, 2].plot(wv_Clementine, radf_Clementine)
    ax_30_0_30[0, 2].plot(wv_global, median_filter(alb, 11))
    ph_hillier100 = hillier1999(wv_hillier, alpha_epoch, mare=True)
    ph_hillier30 = hillier1999(wv_hillier, 30, mare=True)
    if id_num == 0:
        ph_hillier100Interpolation = hillier1999(wv_Clementine, alpha_epoch, mare=True)
        ph_hillier30Interpolation = hillier1999(wv_Clementine, 30, mare=True)
        ax_Clementine.plot(wv_hillier, ph_hillier100, "x", c="grey")
        ax_Clementine.plot(wv_hillier, ph_hillier30, "v", c="k")
        ax_Clementine.plot(wv_Clementine.data, ph_hillier100Interpolation, ls="--", c="grey")
        ax_Clementine.plot(wv_Clementine.data, ph_hillier30Interpolation, ls="--", c="k")
        ax_ratio_Clementine.plot(wv_Clementine.data, ph_hillier30Interpolation / ph_hillier100Interpolation,
                                 c="r", ls="--", alpha=.6)
        ax_Clementine.annotate(f"Clementine", (1_900, 0.05), bbox=bbox_radf)
        ax_Clementine.annotate(f"Hillier + LS", (1_910, 0.035))
        ax_Clementine.annotate(f"$\\alpha \\in (0, 85)°$", (1_910, 0.025))

    max_i = np.max([np.max(ph_buratti30), np.max(ph_hicks30), np.max(ph_hillier30)])
    min_i = np.min([np.min(ph_buratti100), np.min(ph_hicks100), np.min(ph_hillier100)])
    max_radf_i = np.max([np.max(radf_ROLO), np.max(radf_M3), np.max(radf_Clementine)])
    min_radf_i = np.min([np.min(radf_ROLO), np.min(radf_M3), np.min(radf_Clementine)])

    if id_num == 0:
        max_axtwinx = max_i
        min_axtwinx = min_i
        max_radf = max_radf_i
        min_radf = min_radf_i
    else:
        if max_axtwinx < max_i:
            max_axtwinx = max_i
        if min_axtwinx > min_i:
            min_axtwinx = min_i
        if max_radf < max_radf_i:
            max_radf = max_radf_i
        if min_radf > min_radf_i:
            min_radf = min_radf_i

ax_ROLO.set_xlim(np.min(wv_global.data), np.max(wv_global.data))
ax_M3.set_xlim(np.min(wv_global.data), np.max(wv_global.data))
ax_Clementine.set_xlim(np.min(wv_global.data), np.max(wv_global.data))
ax_ROLO.set_ylim(min_axtwinx, max_axtwinx)
ax_M3.set_ylim(min_axtwinx, max_axtwinx)
ax_Clementine.set_ylim(min_axtwinx, max_axtwinx)
ax_ROLO.set_yscale("log")
ax_M3.set_yscale("log")
ax_Clementine.set_yscale("log")
ax_ROLO.set_yticks([])
ax_M3.set_yticks([])
ax_30_0_30[0, 1].set_yticks([])
ax_30_0_30[0, 2].set_yticks([])
ax_30_0_30[1, 1].set_yticks([])
ax_30_0_30[1, 2].set_yticks([])
ax_ratio_ROLO.set_yticks([])
ax_ratio_M3.set_yticks([])
ax_ratio_Clementine.set_yticks([])
min_upper = np.min([np.min(median_filter(arr.data[~arr.mask], 11)) for arr in ALBGLOBAL_MARE[0:8]])
max_upper = np.max([np.max(median_filter(arr.data[~arr.mask], 11)) for arr in ALBGLOBAL_MARE[0:8]])
ax_30_0_30[0, 0].set_ylim(min_upper, max_upper)
ax_30_0_30[0, 1].set_ylim(min_upper, max_upper)
ax_30_0_30[0, 2].set_ylim(min_upper, max_upper)
ax_30_0_30[1, 0].set_ylim(min_radf, max_radf)
ax_30_0_30[1, 1].set_ylim(min_radf, max_radf)
ax_30_0_30[1, 2].set_ylim(min_radf, max_radf)
ax_30_0_30[0, 0].grid()
ax_30_0_30[0, 1].grid()
ax_30_0_30[0, 2].grid()
ax_30_0_30[1, 0].grid()
ax_30_0_30[1, 1].grid()
ax_30_0_30[1, 2].grid()
ax_M3.legend(loc="upper left")
ax_30_0_30[1, 1].legend(loc="upper center")
ax_ratio_M3.legend(loc="best")
ax_30_0_30[1, 1].set_xlabel(f"$\lambda$ (nm)", fontsize=15)
ax_30_0_30[0, 0].set_ylabel(r'$A(\lambda, \alpha, i, e)$')
ax_30_0_30[1, 0].set_ylabel(r'$RADF(30°, 0°, 30°)$')
ax_Clementine.set_ylabel(f'$f(\\alpha)$')
fig_30_0_30.show()


# HL
# fig_30_0_30HL, ax_30_0_30HL = plt.subplots(1, 1)
# for id_num, (wv_global, alb, alb_err, date, ph_coo) in enumerate(zip(WVGLOBAL_HL[0:8], ALBBROADGLOBAL_HL[0:8],
#                                                                      ALBBROADGLOBALERR_HL[0:8], DATESGLOBAL_HL[0:8],
#                                                                      hl_PHcoo[0:8])):
#     lo_i, la_i = ph_coo
#     radf_cali = radf30_0_30(alb, 100, la_i, lo_i)
#     ax_30_0_30HL.plot(wv_global, median_filter(radf_cali, 11), label=f"Site: {id_num+1}")
# ax_30_0_30HL.grid()
# ax_30_0_30HL.legend()
# ax_30_0_30HL.set_xlabel(f"$\lambda$ (nm)")
# ax_30_0_30HL.set_ylabel(r'$RADF(30°, 0°, 30°)$')
# fig_30_0_30HL.suptitle("HIGHLANDS DALE CULIAOOOOO")
# fig_30_0_30HL.show()

# fig_errHL, ax_errHL = plt.subplots(1, 1)
# for id_num, (wv_global, alb, alb_err, date, ph_coo) in enumerate(zip(WVGLOBAL_HL[0:8], ALBBROADGLOBAL_HL[0:8],
#                                                                      ALBBROADGLOBALERR_HL[0:8], DATESGLOBAL_HL[0:8],
#                                                                      hl_PHcoo[0:8])):
#     ax_errHL.plot(wv_global, alb_err, label=f"Site: {id_num+1}")
# ax_errHL.set_yscale("log")
# ax_errHL.grid()
# ax_errHL.legend()
# ax_errHL.set_xlabel(f"$\lambda$ (nm)")
# ax_errHL.set_xlim(650, 700)
# fig_errHL.show()


# fig_eq, ax_eq = plt.subplots(1, 1)
# for id_num, (wv_global, alb, alb_err, date, ph_coo) in enumerate(zip(WVGLOBAL_MARE[0:8], ALBBROADGLOBAL_MARE[0:8],
#                                                                      ALBBROADGLOBALERR_MARE[0:8], DATESGLOBAL_MARE[0:8],
#                                                                      mare_PHcoo[0:8])):
#     lo_i, la_i = ph_coo
#     a_eq = eq_albedo(alb, 100, la_i, lo_i, eta=.34)
#     ax_eq.plot(wv_global, median_filter(a_eq, 11), label=f"Site: {id_num+1}")
# ax_eq.grid()
# ax_eq.legend()
# ax_eq.set_xlabel(f"$\lambda$ (nm)")
# ax_eq.set_ylabel(r'$A_{eq}(\lambda, \alpha)$')
# fig_eq.show()

# fig_FE, ax_FE = plt.subplots(1, 1)
# for id_num, (wv_global, alb, alb_err, date) in enumerate(zip(WVGLOBAL_MARE[0:8], ALBBROADGLOBAL_MARE[0:8],
#                                                              ALBBROADGLOBALERR_MARE[0:8], DATESGLOBAL_MARE[0:8])):
#     Fe_i, wtFeO_i, R757, R918_ratio_757 = fe_idx(wv_global, alb, deltawv1=1, deltawv2=1, show=False)
#     ax_FE.plot(R757, R918_ratio_757, 'o', label=f"Site: {id_num+1}")
# ax_FE.set_xlabel(r'$R_{757}$')
# ax_FE.set_ylabel(r'$R_{918}/R_{757}$')
# ax_FE.legend()
# ax_FE.grid()
# fig_FE.show()


#fig_all, ax_all = plt.subplots(figsize=(9, 6))
#for wv_global, alb, date in zip(WVGLOBAL_MARE, ALBBROADGLOBAL_MARE, DATESGLOBAL_MARE):
#    ax_all.plot(wv_global, alb, lw=.5)
#wu2018_path = "/home/yiyo/moon_data_ref/Wu_2018/table2.dat"
#wu2018 = np.loadtxt(wu2018_path)
#ax_wu = ax_all.twinx()
#ax_wu.plot(wu2018[:, 0], wu2018[:, -4], ls=":")
#ax_wu.plot(wu2018[:, 0], wu2018[:, -3], ls=":")
#ax_wu.plot(wu2018[:, 0], wu2018[:, -2], ls=":")
#ax_wu.plot(wu2018[:, 0], wu2018[:, -1], ls=":")
#fig_all.show()

#alb_median = np.ma.masked_array(np.ma.median(ALBBROADGLOBAL_MARE, axis=0).data, mask=WVGLOBAL_MARE[0].mask)
#fig_ratio, ax_ratio = plt.subplots()
#for wv_global, alb, date, in zip(WVGLOBAL_MARE, ALBBROADGLOBAL_MARE, DATESGLOBAL_MARE):
#    alb_ratio = spectral_ratio(wv_global, alb, alb_median, [530, 550])
#    ax_ratio.plot(wv_global, alb_ratio, lw=.5)
#    ax_ratio.set_xlim(np.min(wv_global.data), 1_000)
#fig_ratio.show()


"""
fig_general, ax_general = plt.subplots(1, 1, figsize=(10, 5), layout="constrained")
min_alb_list, max_alb_list = check_lims(WVGLOBAL_MARE, ALBGLOBAL_MARE, lims=[600, 2_500], list=True)
#min_alb_list, max_alb_list = check_lims(WVGLOBAL_MARE_0, ALBGLOBAL_MARE_0, lims=[700, 1_750], list=True)
range_to_norm = [[1_540, 1_560], [720, 750]]
ALBFLAT = []
ALBFLATERR = []
# Corrected edge
for wv_global, alb_global, alb_br_global, alb_br_err_global, \
       date_global, mi_a, ma_a, ax_g in zip(WVGLOBAL_MARE, ALBGLOBAL_MARE, ALBBROADGLOBAL_MARE, ALBBROADGLOBALERR_MARE,
                                             DATESGLOBAL_MARE, min_alb_list, max_alb_list, fig_general.axes):
#for wv_global, alb_global, alb_br_global, alb_br_err_global, \
#       date_global, mi_a, ma_a, ax_g in zip(WVGLOBAL_MARE_0, ALBGLOBAL_MARE_0, ALBBROADGLOBAL_MARE_0, ALBBROADGLOBALERR_MARE_0,
#                                             DATESGLOBAL_MARE_0, min_alb_list, max_alb_list, fig_general.axes):
    ax_g.grid()
    ax_g.set_ylabel('Reflectance')
    ax_g.set_xlabel(r'$\lambda$ (nm)')
    alb_flat, alb_flat_err = flat_reflectance(wv_global, alb_br_global, alb_br_err_global, range_to_norm,
                                              n_poly_flat=1, ax_toFit=ax_g)
    ALBFLAT.append(alb_flat)
    ALBFLATERR.append(alb_flat_err)
    mask_band = np.logical_and(wv_global.data > 600, wv_global.data < 2_500)
    ax_g.plot(wv_global, alb_br_global)
    ax_g.plot(wv_global, median_filter(alb_br_global, 11), ls="--", c="k")
    ax_g.plot(wv_global, alb_br_global / alb_flat, ls="--", c="r")
    ax_g.set_xlim(600, 2_500)
    ax_g.set_ylim(mi_a, ma_a)
    ax_g.set_title(date_global[0])
    #ax_g.set_yscale("log")
fig_general.show()

fig_fit, ax_fit = plt.subplots(figsize=(15, 13), layout="constrained")
min_albflat, max_albflat = check_lims(WVGLOBAL_MARE, ALBFLAT, lims=[550, 2_150], max_mode="median", s_max=2)
# min_albflat, max_albflat = check_lims(WVGLOBAL_MARE_0, ALBFLAT, lims=[550, 2_150], max_mode="median", s_max=2)
min_std, max_std = check_lims(WVGLOBAL_MARE, ALBFLATERR, lims=[550, 2_150])
# min_std, max_std = check_lims(WVGLOBAL_MARE_0, ALBFLATERR, lims=[550, 2_150])
subfigs_fit = fig_fit.subfigures(3, 3, wspace=0.01, width_ratios=[1, 1, 1])
for wv_global, alb_flat, alb_flat_err, date_global, \
          sf_f in zip(WVGLOBAL_MARE, ALBFLAT, ALBFLATERR, DATESGLOBAL_MARE, subfigs_fit.flatten()):
#for wv_global, alb_flat, alb_flat_err, date_global, \
          #sf_f in zip(WVGLOBAL_MARE_0, ALBFLAT, ALBFLATERR, DATESGLOBAL_MARE_0, subfigs_fit.flatten()):
    wv_tell, alFlat_tell = return_telluricData(wv_global, alb_flat)
    axs_sf = sf_f.subplots(3, 1, height_ratios=[1, 2, 1], sharex=True)
    # plot the limits to norm
    for r in range_to_norm:
        axs_sf[1].axvspan(r[0], r[1], alpha=.3, color="r")
    axs_sf[0].set_title(date_global[0])
    axs_sf[1].set_ylabel("Reflectance normalized")
    axs_sf[2].set_xlabel(r'$\lambda$ (nm)')
    axs_sf[0].grid()
    axs_sf[1].grid()
    axs_sf[2].grid()
    # mask_band = np.logical_and(wv_global.data > 700, wv_global.data < 1_750)
    mask_band = np.logical_and(wv_global.data > np.min(range_to_norm), wv_global.data < np.max(range_to_norm))
    #axs_sf[0].plot(wv_global, alb_flat, lw=.5)
    axs_sf[1].plot(wv_global, median_filter(alb_flat, 11))
    axs_sf[1].plot(wv_tell, alFlat_tell, lw=.5, c="grey", alpha=.5)
    print(wv_tell.data[~wv_tell.mask])
    print(alFlat_tell.data[~alFlat_tell.mask])
    axs_sf[0].plot(wv_global, alb_flat_err*np.sqrt(np.pi*(2*11 + 1)/(4*11)), lw=.5, c="k")
    axs_sf[0].set_ylabel(r'$\sigma_{\lambda}$')
    # data to fit
    wv_maskedArr_toFit = np.ma.masked_array(wv_global.data, mask=np.logical_or(wv_global.mask, ~mask_band))
    wv_toFit = wv_global.data[~wv_maskedArr_toFit.mask]
    alb_flat_toFit = alb_flat.data[~wv_maskedArr_toFit.mask]
    alb_flat_err_toFit = alb_flat_err.data[~wv_maskedArr_toFit.mask]
    p0 = -.02, -.1, -.02, 850, 1_050, 1_200, .5, .5, .5,
    #popt, pcov = curve_fit(minimize_function, wv_toFit, alb_flat_toFit, p0=p0, sigma=alb_flat_err_toFit,
    #                       bounds=((-.2, -.2, -.2, 800, 950, 1_150, 0, 0, 0),
    #                               (-.02, -.05, -.02, 900, 1_080, 1_250, 1, 1, 1)))
    alb_err_medianFilter = alb_flat_err_toFit*np.sqrt(np.pi*(2*11 + 1)/(4*11))
    popt, pcov = curve_fit(minimize_function, wv_toFit, median_filter(alb_flat_toFit, 11), p0=p0, sigma=alb_err_medianFilter,
                          bounds=((-.2, -.2, -.2, 810, 950, 1_200, 0, 0, 0),
                                  (-.02, -.05, -.02, 900, 1_080, 1_250, 1, 1, 1)))
    # print(popt)
    perr = np.sqrt(np.diag(pcov))
    popt_mgm1 = [popt[0], popt[3], popt[6]]
    popt_mgm2 = [popt[1], popt[4], popt[7]]
    popt_mgm3 = [popt[2], popt[5], popt[8]]
    axs_sf[1].plot(wv_global.data, mgm(popt, wv_global.data), ls="--")
    axs_sf[1].plot(wv_global.data, mgm_individual(popt_mgm1, wv_global.data), lw=.7)
    axs_sf[1].plot(wv_global.data, mgm_individual(popt_mgm2, wv_global.data), lw=.7)
    axs_sf[1].plot(wv_global.data, mgm_individual(popt_mgm3, wv_global.data), lw=.7)
    axs_sf[1].axvline(popt[3], ls="--", c="g", label=f"Band center: {popt[3]:.0f} +/- {perr[3]:.1f} nm, "
                                                     f"Abs: {popt[0]:.3f} +/- {perr[0]:.3f}")
    axs_sf[1].axvline(popt[4], ls="--", c="g", label=f"Band center: {popt[4]:.0f} +/- {perr[4]:.1f} nm, "
                                                     f"Abs: {popt[1]:.3f} +/- {perr[1]:.3f}")
    axs_sf[1].axvline(popt[5], ls="--", c="g", label=f"Band center: {popt[5]:.0f} +/- {perr[5]:.1f} nm, "
                                                     f"Abs: {popt[2]:.3f} +/- {perr[2]:.3f}")
    axs_sf[1].legend(fontsize=8)
    ###########################################################################
    axs_sf[2].set_ylabel(r'$y - y_{model}$')
    axs_sf[2].plot(wv_global, alb_flat - mgm(popt, wv_global), lw=.5, c="grey")
    # axs_sf[1].plot(wv_global, )
    #axs_sf[0].set_xlim(700, 1_750)
    # axs_sf[1].set_xlim(700, 1_750)
    axs_sf[1].set_xlim(700, 2_200)
    #axs_sf[1].set_ylim(min_albflat, max_albflat)
    axs_sf[1].set_ylim(.88, 1.05)
    axs_sf[2].set_xlim(700, 2_200)
    axs_sf[0].set_xlim(700, 2_200)
    axs_sf[0].set_ylim(min_std, max_std)
    axs_sf[2].set_ylim(-.1, .1)

fig_fit.show()

fig_toNorm, ax_toNorm = plt.subplots(figsize=(7, 3), layout="constrained")
# min_albflat, max_albflat = check_lims(WVGLOBAL, ALBFLAT, lims=[700, 1_750], max_mode="median", s_max=2)
# min_std, max_std = check_lims(WVGLOBAL, ALBFLATERR, lims=[700, 1_750])
# subfigs_fit = fig_fit.subfigures(2, 2, wspace=0.01, width_ratios=[1, 1])
ALBGLOBAL_750nm = []
for idx, (wv_global, alb_global, date_global,
          sf_f) in enumerate(zip(WVGLOBAL_MARE, ALBGLOBAL_MARE, DATESGLOBAL_MARE, subfigs_fit.flatten())):
#for idx, (wv_global, alb_global, date_global, \
#           sf_f) in enumerate(zip(WVGLOBAL_MARE_0, ALBGLOBAL_MARE_0, DATESGLOBAL_MARE_0, subfigs_fit.flatten())):
    al_normalized = normalizeXXnm_v2(alb_global, wv_global, [740, 760])
    ALBGLOBAL_750nm.append(al_normalized)
    #ax_toNorm.plot(wv_global, al_normalized, label=f"Site {idx+1}", lw=.5)
    ax_toNorm.plot(wv_global, median_filter(al_normalized, 11), label=f"Site {idx+1}")
    wv_tell, al_tell = return_telluricData(wv_global, al_normalized)
    ax_toNorm.plot(wv_tell, al_tell, color="grey", alpha=.2, lw=.5)
ax_toNorm.grid()
ax_toNorm.legend()
min_albnorm, max_albnorm = check_lims(WVGLOBAL_MARE, ALBGLOBAL_750nm, lims=[550, 2_150], max_mode="median", s_max=3)
#min_albnorm, max_albnorm = check_lims(WVGLOBAL_MARE_0, ALBGLOBAL_750nm, lims=[550, 2_150], max_mode="median", s_max=3)
ax_toNorm.set_xlim(550, 2_150)
ax_toNorm.set_ylim(min_albnorm, max_albnorm)
ax_toNorm.set_ylabel("Reflectance normalized at 750 nm")
ax_toNorm.set_xlabel(r'$\lambda$ (nm)')
fig_toNorm.show()
"""