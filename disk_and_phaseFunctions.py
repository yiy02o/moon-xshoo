import numpy as np
import matplotlib.pyplot as plt
import scipy
from velikodsky_spectrum import *


################################################ DISK FUNCTIONS #####################################################
def akimov_disk(lo_rad, la_rad, alpha_rad, eta):
    left_term_diskAk = (np.cos(la_rad)) ** (eta * alpha_rad / (np.pi - alpha_rad))
    mid_term_diskAk = np.cos(alpha_rad / 2) / np.cos(lo_rad)
    right_term_diskAk = np.cos((lo_rad - alpha_rad / 2) * np.pi / (np.pi - alpha_rad))
    Xl = left_term_diskAk * mid_term_diskAk * right_term_diskAk
    return Xl


def LS_disk(i_rad, e_rad):
    Xl = np.cos(i_rad) / (np.cos(i_rad) + np.cos(e_rad))
    return Xl


def mcEwen_disk(i_rad, e_rad, alpha_deg):
    A, B, C = -.019, .242*1e-3, -1.46*1e-6
    L_alpha = 1 + A*alpha_deg + B*alpha_deg**2 + C*alpha_deg**3
    Xl = 2*L_alpha*(np.cos(i_rad) / (np.cos(i_rad) + np.cos(e_rad))) + (1 - L_alpha)*np.cos(i_rad)
    return Xl


################################################ PHASE FUNCTIONS #####################################################


def buratti2011(wv_arr, alpha_deg, mare=True, method_interp="univariate", bc_type="natural", s=0.5):
    wv_arr = wv_arr.data if isinstance(wv_arr, np.ma.MaskedArray) else wv_arr
    if method_interp != "CS" and method_interp != "interp1d" and method_interp != "univariate":
        raise ValueError("Choose a valid interpolator.")

    def ph_function(alpha, params):
        C0, C1, A0, A1, A2, A3, A4 = params
        C0, C1, A0, A1, A2, A3, A4 = C0*1e-2, C1, A0, A1*1e-2, A2*1e-4, A3*1e-6, A4*1e-8
        return C0*np.exp(-C1*alpha) + A0 + A1*alpha + A2*alpha**2 + A3*alpha**3 + A4*alpha**4

    if mare:
        txtToNumpy = np.loadtxt("/home/yiyo/moon_data_ref/ph_functions/buratti2011/buratti2011_mare.txt",
                                skiprows=4, usecols=[0, 2, 3, 4, 5, 6, 7, 8])
        f_model = np.array([ph_function(alpha_deg, row[1:]) for row in txtToNumpy])

        # interpolation
        wv_sample, nonrepeat_idx = np.unique(txtToNumpy[:, 0], return_index=True)
        f_model = f_model[nonrepeat_idx]
        if method_interp == "CS":
            ph_mareFunc_interp = scipy.interpolate.CubicSpline(wv_sample, f_model, bc_type=bc_type)
        if method_interp == "interp1d":
            ph_mareFunc_interp = scipy.interpolate.interp1d(wv_sample, f_model, kind="slinear")
        if method_interp == "univariate":
            ph_mareFunc_interp = scipy.interpolate.UnivariateSpline(wv_sample, f_model, s=s)
        ph_mare_func = ph_mareFunc_interp(wv_arr)
        return ph_mare_func
    else:
        txtToNumpy = np.loadtxt("/home/yiyo/moon_data_ref/ph_functions/buratti2011/buratti2011_highlands.txt",
                                skiprows=4, usecols=[0, 2, 3, 4, 5, 6, 7, 8])
        f_model = np.array([ph_function(alpha_deg, row[1:]) for row in txtToNumpy])

        # interpolation
        wv_sample, nonrepeat_idx = np.unique(txtToNumpy[:, 0], return_index=True)
        f_model = f_model[nonrepeat_idx]
        if method_interp == "CS":
            ph_highlandsFunc_interp = scipy.interpolate.CubicSpline(wv_sample, f_model, bc_type=bc_type)
        if method_interp == "interp1d":
            ph_highlandsFunc_interp = scipy.interpolate.interp1d(wv_sample, f_model, kind="slinear")
        if method_interp == "univariate":
            ph_highlandsFunc_interp = scipy.interpolate.UnivariateSpline(wv_sample, f_model, s=s)
        ph_highlands_func = ph_highlandsFunc_interp(wv_arr)
        return ph_highlands_func


def hicks2011(wv_arr, alpha_deg, mare=True, method_interp="univariate", bc_type="natural", s=0.5):
    wv_arr = wv_arr.data if isinstance(wv_arr, np.ma.MaskedArray) else wv_arr
    if method_interp != "CS" and method_interp != "interp1d" and method_interp != "univariate":
        raise ValueError("Choose a valid interpolator.")

    def ph_function(alpha, params):
        A0, A1, A2, A3, A4, A5, A6 = params
        A0, A1, A2, A3, A4, A5, A6 = A0, A1*1e-2, A2*1e-4, A3*1e-6, A4*1e-8, A5*1e-10, A6*1e-12
        return A0 + A1*alpha + A2*alpha**2 + A3*alpha**3 + A4*alpha**4 + A5*alpha**5 + A6*alpha**6

    if mare:
        txtToNumpy = np.loadtxt("/home/yiyo/moon_data_ref/ph_functions/hicks2011/hicksMare_phfunction.txt",
                                skiprows=4)
        f_model = np.array([ph_function(alpha_deg, row[2:]) for row in txtToNumpy])

        # interpolation
        if method_interp == "CS":
            ph_mareFunc_interp = scipy.interpolate.CubicSpline(txtToNumpy[:, 1], f_model, bc_type=bc_type)
        if method_interp == "interp1d":
            ph_mareFunc_interp = scipy.interpolate.interp1d(txtToNumpy[:, 1], f_model, kind="slinear")
        if method_interp == "univariate":
            ph_mareFunc_interp = scipy.interpolate.UnivariateSpline(txtToNumpy[:, 1], f_model, s=s)
        ph_mare_data = ph_mareFunc_interp(wv_arr)
        return ph_mare_data
    else:
        txtToNumpy = np.loadtxt("/home/yiyo/moon_data_ref/ph_functions/hicks2011/hicksNotMare_phfunction.txt",
                                skiprows=4)
        f_model = np.array([ph_function(alpha_deg, row[2:]) for row in txtToNumpy])

        # interpolation
        if method_interp == "CS":
            ph_notmareFunc_interp = scipy.interpolate.CubicSpline(txtToNumpy[:, 1], f_model, bc_type=bc_type)
        if method_interp == "interp1d":
            ph_notmareFunc_interp = scipy.interpolate.interp1d(txtToNumpy[:, 1], f_model, kind="slinear")
        if method_interp == "univariate":
            ph_notmareFunc_interp = scipy.interpolate.UnivariateSpline(txtToNumpy[:, 1], f_model, s=s)
        ph_notmare_data = ph_notmareFunc_interp(wv_arr)
        return ph_notmare_data


def hillier1999(wv_arr, alpha_deg, mare=True, method_interp="univariate", bc_type="natural", s=0.5):
    wv_arr = wv_arr.data if isinstance(wv_arr, np.ma.MaskedArray) else wv_arr
    table_hillier_mare = np.array([[415,   -.0198, .6,   .226, -11.08, 30.82, -39.25, 17.89],
                                   [750,   -.0661, .359, .362, -20.01, 61.78, -81.46, 37.16],
                                   [900,   -.0633, .356, .366, -19.76, 60.27, -78.78, 35.63],
                                   [950,   -.0558, .373, .358, -18.73, 55.84, -71.52, 31.61],
                                   [1_000, -.0486, .320, .328, -15.23, 44.10, -56.31, 24.76]])

    table_hillier_hl = np.array([[415,   .1053, .541, .316, -9.65,  23.57, -37.46,  24.18],
                                 [750,   .1718, .374, .414, -4.48,  -7.42,  18.75,  -9.26],
                                 [900,   .1598, .450, .451, -6.72,   3.81,  -3.47,   5.36],
                                 [950,   .1589, .498, .461, -7.50,   7.44,  -9.37,   8.42],
                                 [1_000, .3545, .194, .193, 19.80, -89.65, 136.01, -69.15]])
    if method_interp != "CS" and method_interp != "interp1d" and method_interp != "univariate":
        raise ValueError("Choose a valid interpolator.")

    def ph_function(alpha, params):
        b0, b1, a0, a1, a2, a3, a4 = params
        b0, b1, a0, a1, a2, a3, a4 = b0, b1, a0, a1*1e-3, a2*1e-5, a3*1e-7, a4*1e-9
        return b0*np.exp(-b1*alpha) + a0 + a1*alpha + a2*alpha**2 + a3*alpha**3 + a4*alpha**4

    if mare:
        f_model = np.array([ph_function(alpha_deg, row[1:]) for row in table_hillier_mare])

        # interpolation
        if method_interp == "CS":
            ph_mareFunc_interp = scipy.interpolate.CubicSpline(table_hillier_mare[:, 0], f_model, bc_type=bc_type)
        if method_interp == "interp1d":
            ph_mareFunc_interp = scipy.interpolate.interp1d(table_hillier_mare[:, 0], f_model, kind="slinear")
        if method_interp == "univariate":
            ph_mareFunc_interp = scipy.interpolate.UnivariateSpline(table_hillier_mare[:, 0], f_model, s=s)
        ph_mare_data = ph_mareFunc_interp(wv_arr)
        return ph_mare_data
    else:
        f_model = np.array([ph_function(alpha_deg, row[1:]) for row in table_hillier_hl])

        # interpolation
        if method_interp == "CS":
            ph_notmareFunc_interp = scipy.interpolate.CubicSpline(table_hillier_hl[:, 0], f_model, bc_type=bc_type)
        if method_interp == "interp1d":
            ph_notmareFunc_interp = scipy.interpolate.interp1d(table_hillier_hl[:, 0], f_model, kind="slinear")
        if method_interp == "univariate":
            ph_notmareFunc_interp = scipy.interpolate.UnivariateSpline(table_hillier_hl[:, 0], f_model, s=s)
        ph_notmare_data = ph_notmareFunc_interp(wv_arr)
        return ph_notmare_data


def korokhin2007(wv_arr, alpha_deg, method_interp="univariate", bc_type="natural", s=0.5):
    wv_arr = wv_arr.data if isinstance(wv_arr, np.ma.MaskedArray) else wv_arr
    table_korokhin = np.array([[359,     0.346, 3.457,  0.817, 0.038],
                               [361.5,   0.358, 3.169,  0.800, 0.041],
                               [392.6,   0.299, 3.974,  0.864, 0.037],
                               [415.5,   0.354, 3.821,  0.821, 0.032],
                               [440.0,   0.291, 4.215,  0.875, 0.031],
                               [457.3,   0.317, 4.577,  0.865, 0.027],
                               [501.2,   0.308, 4.862,  0.877, 0.027],
                               [548.0,   0.264, 6.036,  0.925, 0.028],
                               [626.4,   0.543, 9.104,  0.851, 0.038],
                               [729.7,   0.379, 8.277,  0.905, 0.036],
                               [859.5,   0.291, 9.263,  0.957, 0.030],
                               [1_063.5, 0.412, 11.473, 0.943, 0.039]])
    if method_interp != "CS" and method_interp != "interp1d" and method_interp != "univariate":
        raise ValueError("Choose a valid interpolator.")

    def ph_function(alpha, params):
        m1, rho, m2, sigma = params
        return m1 * np.exp(-rho * alpha) + m2 * np.exp(-0.7 * alpha)

    alpha_rad = alpha_deg * np.pi / 180
    f_model = np.array([ph_function(alpha_rad, row[1:]) for row in table_korokhin])

    # interpolation
    if method_interp == "CS":
        ph_Func_interp = scipy.interpolate.CubicSpline(table_korokhin[:, 0], f_model, bc_type=bc_type)
    if method_interp == "interp1d":
        ph_Func_interp = scipy.interpolate.interp1d(table_korokhin[:, 0], f_model, kind="slinear")
    if method_interp == "univariate":
        ph_Func_interp = scipy.interpolate.UnivariateSpline(table_korokhin[:, 0], f_model, s=s)
    ph_data = ph_Func_interp(wv_arr)
    return ph_data


def velikodsky2011(wv_arr, alpha_deg, terrain="mare", method_interp="univariate", bc_type="natural", s=0.5,
                   mask_fmodel=None):
    """This function only works for phase angle at the range of > 55°, far as I understand."""
    if method_interp != "CS" and method_interp != "interp1d" and method_interp != "univariate":
        raise ValueError("Choose a valid interpolator.")

    f_model = recover_Aeq(alpha_deg, plot=False)
    # mask x and y
    if mask_fmodel is not None:
        f_model = f_model[mask_fmodel]
        wv_model = velik_table[:, 0][mask_fmodel]
    else:
        wv_model = velik_table[:, 0]

    # Interpolation
    if method_interp == "CS":
        ph_Func_interp = scipy.interpolate.CubicSpline(wv_model, f_model, bc_type=bc_type)
    if method_interp == "interp1d":
        ph_Func_interp = scipy.interpolate.interp1d(wv_model, f_model, kind="slinear")
    if method_interp == "univariate":
        ph_Func_interp = scipy.interpolate.UnivariateSpline(wv_model, f_model, s=s)

    ph_data = ph_Func_interp(wv_arr)
    return ph_data


wv_buratti = np.loadtxt("/home/yiyo/moon_data_ref/ph_functions/buratti2011/buratti2011_mare.txt",
                        skiprows=4, usecols=[0, 2, 3, 4, 5, 6, 7, 8])[:, 0]
wv_hicks = np.loadtxt("/home/yiyo/moon_data_ref/ph_functions/hicks2011/hicksMare_phfunction.txt",
                      skiprows=4)[:, 1]
wv_hillier = np.array([415, 750, 900, 950, 1_000])
wv_korokhin = np.array([359, 361.5, 392.6, 415.5, 440.0, 457.3, 501.2, 548.0, 626.4, 729.7, 859.5, 1_063.5])

fig_test, ax_test = plt.subplots()
fig_test.suptitle("Mare regions", fontsize=25)

# Buratti et al., 2011
ph_buratti100 = buratti2011(wv_buratti, 100, mare=True)
ph_buratti90 = buratti2011(wv_buratti, 90, mare=True)
ph_buratti0 = buratti2011(wv_buratti, 0, mare=True)
ax_test.plot(wv_buratti, ph_buratti100, ls=":", label=f"Buratti et al., 2011, $\\alpha = 100°$", c="grey")
ax_test.plot(wv_buratti, ph_buratti90, ls=":", c="b", alpha=.8)
ax_test.plot(wv_buratti, ph_buratti0, ls=":", c="r")
# ax_test.annotate(r'$\alpha = 100°$', (wv_buratti[-1], ph_buratti100[-1] - .03), c="grey")
ax_test.annotate(r'$\alpha = 90°$', (wv_buratti[-1], ph_buratti90[-1] + .01), c="b")
ax_test.annotate(r'$\alpha = 0°$', (wv_buratti[-1] - 100, ph_buratti0[-1] + .02), c="r")

# Hicks et al., 2011
ph_hicks100 = hicks2011(wv_hicks, 100, mare=True)
ph_hicks80 = hicks2011(wv_hicks, 80, mare=True)
ph_hicks0 = hicks2011(wv_hicks, 0, mare=True)
ax_test.plot(wv_hicks, ph_hicks100, c="grey", alpha=.8, label=f"Hicks et al., 2011, $\\alpha = 100°$")
ax_test.plot(wv_hicks, ph_hicks80, c="b")
ax_test.plot(wv_hicks, ph_hicks0, c="r")
# ax_test.annotate(r'$\alpha = 100°$', (wv_hicks[-1] + 50, ph_hicks100[-1]), c="grey")
ax_test.annotate(r'$\alpha = 80°$', (wv_hicks[-1] + 50, ph_hicks80[-1]), c="b")
ax_test.annotate(r'$\alpha = 0°$', (wv_hicks[-1] + 50, ph_hicks0[-1]), c="r")

# Hillier et al., 1999
ph_hillier100 = hillier1999(wv_hillier, 100, mare=True)
ph_hillier85 = hillier1999(wv_hillier, 85, mare=True)
ph_hillier0 = hillier1999(wv_hillier, 0, mare=True)
ax_test.plot(wv_hillier, ph_hillier100, "*", label=f"Hillier et al., 1999, $\\alpha = 100°$", c="grey")
ax_test.plot(wv_hillier, ph_hillier85, "*", c="b")
ax_test.plot(wv_hillier, ph_hillier0, "*", c="r")
# ax_test.annotate(r'$\alpha = 100°$', (wv_hillier[-1] + 250, ph_hillier100[-1] - .01), c="grey")
ax_test.annotate(r'$\alpha = 85°$', (wv_hillier[-3], ph_hillier85[-3] - 0.05), c="b")
ax_test.annotate(r'$\alpha = 0°$', (wv_hillier[-3], ph_hillier0[-3] + .02), c="r")

ax_test.axvspan(350, 2_500, color="grey", alpha=.15, label="X-Shooter wavelength coverage")
ax_test.grid()
ax_test.legend()
ax_test.set_xlabel(r'$\lambda$ (nm)')
ax_test.set_ylabel(r'$A_{eq}(\alpha, \lambda)$')
fig_test.show()

# Disk function test
fig_disk, ax_disk = plt.subplots()

lo_to_test = 340
la_to_test = -25
fig_disk.suptitle(f"$\gamma = {la_to_test}$°, $\\beta = {lo_to_test}°$", fontsize=20)

alpha100 = 100
alpha0 = 0
lo_to_diskFunc = (360 - lo_to_test) * np.pi / 180
la_to_diskFunc = la_to_test * np.pi / 180
alpha100_to_diskFunc = alpha100 * np.pi / 180
alpha0_to_diskFunc = alpha0 * np.pi / 180
alpha_sample = np.linspace(0, 120)
alpha_sample_to_rad = alpha_sample * np.pi / 180
i_to_diskFunc100 = np.arccos(np.cos(la_to_diskFunc) * np.cos(alpha100_to_diskFunc - lo_to_diskFunc))
e_to_diskFunc100 = np.arccos(np.cos(la_to_diskFunc) * np.cos(lo_to_diskFunc))
i_to_diskFunc0 = np.arccos(np.cos(la_to_diskFunc) * np.cos(alpha0_to_diskFunc - lo_to_diskFunc))
e_to_diskFunc0 = np.arccos(np.cos(la_to_diskFunc) * np.cos(lo_to_diskFunc))
i_sample_to_rad = np.arccos(np.cos(la_to_diskFunc) * np.cos(alpha_sample_to_rad - lo_to_diskFunc))
e_sample_to_rad = np.arccos(np.cos(la_to_diskFunc) * np.cos(lo_to_diskFunc))

# Akimov
disk_akimov = akimov_disk(lo_to_diskFunc, la_to_diskFunc, alpha_sample_to_rad, .34)
disk_akimov0 = akimov_disk(lo_to_diskFunc, la_to_diskFunc, alpha0_to_diskFunc, .34)
disk_akimov100 = akimov_disk(lo_to_diskFunc, la_to_diskFunc, alpha100_to_diskFunc, .34)
ax_disk.plot(alpha_sample, disk_akimov, c="b")
ax_disk.plot(0, disk_akimov0, "X", label=r'$\alpha = 0°$', c="k")
ax_disk.plot(100, disk_akimov100, "X", label=r'$\alpha = 100°$', c="r")
dy_akimov = -(.8 - .6)
dx_akimov = 85 - 65
angle_akimov = np.rad2deg(np.arctan2(dy_akimov, dx_akimov))
bbox = dict(boxstyle="round", fc="0.9")
ax_disk.annotate(f"Akimov ($\\eta = 0.34$)", (58, 0.73), transform_rotates_text=True, rotation=angle_akimov, bbox=bbox)

# LS
disk_LS = LS_disk(i_sample_to_rad, e_sample_to_rad)
disk_LS_0 = LS_disk(i_to_diskFunc0, e_to_diskFunc0)
disk_LS_100 = LS_disk(i_to_diskFunc100, e_to_diskFunc100)
ax_disk.plot(alpha_sample, disk_LS, c="b", ls="--")
ax_disk.plot(0, disk_LS_0, "X", c="k")
ax_disk.plot(100, disk_LS_100, "X", c="r")
ax_disk.annotate(f"McEwen", (44, 0.72), transform_rotates_text=True, rotation=angle_akimov, bbox=bbox)

# McEwen
disk_mcEwen = mcEwen_disk(i_sample_to_rad, e_sample_to_rad, alpha_sample)
disk_mcEwen0 = mcEwen_disk(i_to_diskFunc0, e_to_diskFunc0, 0)
disk_mcEwen100 = mcEwen_disk(i_to_diskFunc100, e_to_diskFunc100, 100)
ax_disk.plot(alpha_sample, disk_mcEwen, c="b", ls=":")
ax_disk.plot(0, disk_mcEwen0, "X", c="k")
ax_disk.plot(100, disk_mcEwen100, "X", c="r")
dy_LS = -(.5 - .45)
dx_LS = -(20 - 40)
angle_LS = np.rad2deg(np.arctan2(dy_LS, dx_LS))
ax_disk.annotate(f"Lommel-Seeliger", (20, 0.37), transform_rotates_text=True, rotation=angle_LS, bbox=bbox)

ax_disk.set_ylim(ymin=0)
ax_disk.set_xlabel(r'$\alpha$ (°)')
ax_disk.set_ylabel(r'$D(\alpha, i, e, \lambda)$')
ax_disk.legend()
ax_disk.grid()
fig_disk.show()

# Check the interpolation
fig_interp, ax_interp = plt.subplots(4, 1, layout="constrained")
wv_burattiToEval = np.linspace(wv_buratti[0], wv_buratti[-1], 10_000)
wv_hicksToEval = np.linspace(wv_hicks[0], wv_hicks[-1], 10_000)
wv_hillierToEval = np.linspace(wv_hillier[0], wv_hillier[-1], 10_000)
wv_korokhinToeval = np.linspace(wv_korokhin[0], wv_korokhin[-1], 10_000)
bbox_interp = dict(boxstyle="round", fc="0.9")

# Buratti
ph_buratti100Sample = buratti2011(wv_buratti, 100, mare=True, method_interp="univariate", s=0)
ph_buratti100CSDefault = buratti2011(wv_burattiToEval, 100, mare=True, method_interp="CS", bc_type="not-a-knot")
ph_buratti100CSNatural = buratti2011(wv_burattiToEval, 100, mare=True, method_interp="CS", bc_type="natural")
ph_buratti100interp1d = buratti2011(wv_burattiToEval, 100, mare=True, method_interp="interp1d")
ph_buratti100Univariate0 = buratti2011(wv_burattiToEval, 100, mare=True, method_interp="univariate", s=0)
ph_buratti100Univariate05 = buratti2011(wv_burattiToEval, 100, mare=True, method_interp="univariate", s=0.5)
ax_interp[0].plot(wv_buratti, ph_buratti100Sample, 'v', c='r')
ax_interp[0].plot(wv_burattiToEval, ph_buratti100CSDefault, c='k', ls="--")
ax_interp[0].plot(wv_burattiToEval, ph_buratti100CSNatural, c='m', ls="--")
ax_interp[0].plot(wv_burattiToEval, ph_buratti100interp1d, c='lime', ls="--")
ax_interp[0].plot(wv_burattiToEval, ph_buratti100Univariate0, c='orange', ls="--")
ax_interp[0].plot(wv_burattiToEval, ph_buratti100Univariate05, c='b', ls="--")
ax_interp[0].plot(wv_buratti, ph_buratti100, c='grey', alpha=.4)
ax_interp[0].annotate(f"Buratti 2011", (2_300, 0.05), transform_rotates_text=True, bbox=bbox)
ax_interp[0].grid()

# Hicks
ph_hicks100Sample = hicks2011(wv_hicks, 100, mare=True, method_interp="univariate", s=0)
ph_hicks100CSDefault = hicks2011(wv_hicksToEval, 100, mare=True, method_interp="CS", bc_type="not-a-knot")
ph_hicks100CSNatural = hicks2011(wv_hicksToEval, 100, mare=True, method_interp="CS", bc_type="natural")
ph_hicks100interp1d = hicks2011(wv_hicksToEval, 100, mare=True, method_interp="interp1d")
ph_hicks100Univariate0 = hicks2011(wv_hicksToEval, 100, mare=True, method_interp="univariate", s=0)
ph_hicks100Univariate05 = hicks2011(wv_hicksToEval, 100, mare=True, method_interp="univariate", s=0.5)
ax_interp[1].plot(wv_hicks, ph_hicks100Sample, 'v', c='r')
ax_interp[1].plot(wv_hicksToEval, ph_hicks100CSDefault, c='k', ls="--")
ax_interp[1].plot(wv_hicksToEval, ph_hicks100CSNatural, c='m', ls="--")
ax_interp[1].plot(wv_hicksToEval, ph_hicks100interp1d, c='lime', ls="--")
ax_interp[1].plot(wv_hicksToEval, ph_hicks100Univariate0, c='orange', ls="--")
ax_interp[1].plot(wv_hicksToEval, ph_hicks100Univariate05, c='b', ls="--")
ax_interp[1].plot(wv_hicks, ph_hicks100, c='grey', alpha=.4)
ax_interp[1].annotate(f"Hicks 2011", (2_300, 0.2), transform_rotates_text=True, bbox=bbox)
ax_interp[1].grid()

# Hillier
ph_hillier100Sample = hillier1999(wv_hillier, 100, mare=True, method_interp="univariate", s=0)
ph_hillier100CSDefault = hillier1999(wv_hillierToEval, 100, mare=True, method_interp="CS", bc_type="not-a-knot")
ph_hillier100CSNatural = hillier1999(wv_hillierToEval, 100, mare=True, method_interp="CS", bc_type="natural")
ph_hillier100interp1d = hillier1999(wv_hillierToEval, 100, mare=True, method_interp="interp1d")
ph_hillier100Univariate0 = hillier1999(wv_hillierToEval, 100, mare=True, method_interp="univariate", s=0)
ph_hillier100Univariate05 = hillier1999(wv_hillierToEval, 100, mare=True, method_interp="univariate", s=0.5)
ax_interp[2].plot(wv_hillier, ph_hillier100Sample, 'v', c='r')
ax_interp[2].plot(wv_hillierToEval, ph_hillier100CSDefault, c='k', ls="--", label="CS (not-a-knot)")
ax_interp[2].plot(wv_hillierToEval, ph_hillier100CSNatural, c='m', ls="--", label="CS (natural)")
ax_interp[2].plot(wv_hillierToEval, ph_hillier100interp1d, c='lime', ls="--", label="interp1d (slinear)")
ax_interp[2].plot(wv_hillierToEval, ph_hillier100Univariate0, c='orange', ls="--", label=r'Univariate ($s=0$)')
ax_interp[2].plot(wv_hillierToEval, ph_hillier100Univariate05, c='b', ls="--", label=r'Univariate ($s=0.5$)')
ax_interp[2].plot(wv_hillier, ph_hillier100, c='grey', alpha=.4)
ax_interp[2].annotate(f"Hillier 1999", (2_300, 0.075), transform_rotates_text=True, bbox=bbox)
ax_interp[2].grid()

# korokhin
ph_korokhin100 = korokhin2007(wv_korokhin, 100, method_interp="univariate", s=0.)
ph_korokhin100CSDefault = korokhin2007(wv_korokhinToeval, 100, method_interp="CS", bc_type="not-a-knot")
ph_korokhin100CSNatural = korokhin2007(wv_korokhinToeval, 100, method_interp="CS", bc_type="natural")
ph_korokhin100interp1d = korokhin2007(wv_korokhinToeval, 100, method_interp="interp1d")
ph_korokhin100Univariate0 = korokhin2007(wv_korokhinToeval, 100, method_interp="univariate", s=0.)
ph_korokhin100Univariate05 = korokhin2007(wv_korokhinToeval, 100, method_interp="univariate", s=0.5)
ax_interp[3].plot(wv_korokhin, ph_korokhin100, 'v', c='r')
ax_interp[3].plot(wv_korokhinToeval, ph_korokhin100CSDefault, c='k', ls="--")
ax_interp[3].plot(wv_korokhinToeval, ph_korokhin100CSNatural, c='m', ls="--")
ax_interp[3].plot(wv_korokhinToeval, ph_korokhin100interp1d, c='lime', ls="--")
ax_interp[3].plot(wv_korokhinToeval, ph_korokhin100Univariate0, c='orange', ls="--")
ax_interp[3].plot(wv_korokhinToeval, ph_korokhin100Univariate05, c='b', ls="--")
ax_interp[3].plot(wv_korokhin, ph_korokhin100, c='grey', alpha=.4)
ax_interp[3].annotate(f"Korokhin 2007", (2_300, 0.26), transform_rotates_text=True, bbox=bbox)
ax_interp[3].grid()


min_lim_interp = np.min([np.min(wv_burattiToEval), np.min(wv_hicksToEval), np.min(wv_hillierToEval),
                         np.min(wv_korokhinToeval)])
max_lim_interp = np.max([np.max(wv_burattiToEval), np.max(wv_hicksToEval), np.max(wv_hillierToEval),
                         np.max(wv_korokhinToeval)])
ax_interp[0].set_xlim(min_lim_interp, max_lim_interp)
ax_interp[1].set_xlim(min_lim_interp, max_lim_interp)
ax_interp[2].set_xlim(min_lim_interp, max_lim_interp)
ax_interp[3].set_xlim(min_lim_interp, max_lim_interp)
ax_interp[0].set_xticks([])
ax_interp[1].set_xticks([])
ax_interp[2].set_xticks([])
ax_interp[3].set_xlabel(r'$\lambda$ (nm)', fontsize=15)
ax_interp[1].set_ylabel(r'$A_{eq}(\alpha, \lambda)$', fontsize=15)
ax_interp[2].legend(loc="center", fontsize=7)
fig_interp.show()

# phase angles range
phase_angle_range = np.linspace(0, 120, 100)
lamba_eff = [460, 650, 850, 1_200, 1_750, 2_100]
loc_lamba = [(50, .12), (50, .15), (50, .17), (50, .17), (50, .25), (50, .27)]
lamba_buratti_toEval = np.linspace(wv_buratti[0], wv_buratti[-1], 1_000)
lamba_hicks_toEval = np.linspace(wv_hicks[0], wv_hicks[-1], 1_000)
lamba_hillier_toEval = np.linspace(wv_hillier[0], wv_hillier[-1], 1_000)
lamba_korokhin_toEval = np.linspace(wv_korokhin[0], wv_korokhin[-1], 1_000)
dlamba_buratti = 5
dlamba_hicks = 5
dlamba_hillier = 5
dlamba_korokhin = 5
phase_lims_buratti = [0, 90]
phase_lims_hicks = [0, 80]
phase_lims_hillier = [0, 85]
phase_lims_korokhin = [6, 120]
# the good and the bad ones
good_buratti = np.logical_and(phase_angle_range > phase_lims_buratti[0], phase_angle_range < phase_lims_buratti[-1])
good_hicks = np.logical_and(phase_angle_range > phase_lims_hicks[0], phase_angle_range < phase_lims_hicks[-1])
good_hillier = np.logical_and(phase_angle_range > phase_lims_hillier[0], phase_angle_range < phase_lims_hillier[-1])
good_korokhin = np.logical_and(phase_angle_range > phase_lims_korokhin[0], phase_angle_range < phase_lims_korokhin[-1])
# bad phase angles
alphaRange_badBuratti = np.ma.masked_array(phase_angle_range, mask=good_buratti)
alphaRange_badHicks = np.ma.masked_array(phase_angle_range, mask=good_hicks)
alphaRange_badHillier = np.ma.masked_array(phase_angle_range, mask=good_hillier)
alphaRange_badKorokhin = np.ma.masked_array(phase_angle_range, mask=good_korokhin)


def medianXX(wv, al, norm_range):
    mask_range = ~np.logical_and(wv > norm_range[0], wv < norm_range[1])
    y_arr_masked = np.ma.masked_array(al, mask=mask_range)
    median_to_norm = np.ma.median(y_arr_masked)
    return median_to_norm


fig_alpha, ax_alpha = plt.subplots(2, 3, figsize=(12, 5), layout="constrained", sharex=True)


for lamba, ax_i, loc_i in zip(lamba_eff, fig_alpha.axes, loc_lamba):
    # labels
    label_buratti = "Buratti 2011" if lamba == lamba_eff[1] else None
    label_hicks = "Hicks 2011" if lamba == lamba_eff[1] else None
    label_hillier = "Hillier 1999" if lamba == lamba_eff[1] else None
    label_korokhin = "Korokhin 2007" if lamba == lamba_eff[1] else None

    buratti_XXnm = np.asarray([medianXX(lamba_buratti_toEval, buratti2011(lamba_buratti_toEval, alpha_i, mare=True),
                              [lamba - dlamba_buratti, lamba + dlamba_buratti]) for alpha_i in phase_angle_range])
    hicks_XXnm = np.asarray([medianXX(lamba_hicks_toEval, hicks2011(lamba_hicks_toEval, alpha_i, mare=True),
                            [lamba - dlamba_hicks, lamba + dlamba_hicks]) for alpha_i in phase_angle_range])
    hillier_XXnm = np.asarray([medianXX(lamba_hillier_toEval, hillier1999(lamba_hillier_toEval, alpha_i, mare=True),
                              [lamba - dlamba_hillier, lamba + dlamba_hillier]) for alpha_i in phase_angle_range])
    korokhin_XXnm = np.asarray([medianXX(lamba_korokhin_toEval, korokhin2007(lamba_korokhin_toEval, alpha_i),
                               [lamba - dlamba_korokhin, lamba + dlamba_korokhin]) for alpha_i in phase_angle_range])
    # plot them
    ax_i.plot(phase_angle_range[good_buratti], buratti_XXnm[good_buratti], label=label_buratti)
    ax_i.plot(phase_angle_range[good_hicks], hicks_XXnm[good_hicks], label=label_hicks)
    ax_i.plot(phase_angle_range[good_hillier], hillier_XXnm[good_hillier], label=label_hillier)
    ax_i.plot(phase_angle_range[good_korokhin], korokhin_XXnm[good_korokhin], label=label_korokhin)
    # bad ones
    burattiXXnm_bad = np.ma.masked_array(buratti_XXnm, mask=good_buratti)
    hicksXXnm_bad = np.ma.masked_array(hicks_XXnm, mask=good_hicks)
    hillierXXnm_bad = np.ma.masked_array(hillier_XXnm, mask=good_hillier)
    korokhinXXnm_bad = np.ma.masked_array(korokhin_XXnm, mask=good_korokhin)
    ax_i.plot(alphaRange_badBuratti, burattiXXnm_bad, c="grey", alpha=.2, ls="--")
    ax_i.plot(alphaRange_badHicks, hicksXXnm_bad, c="grey", alpha=.2, ls="--")
    ax_i.plot(alphaRange_badHillier, hillierXXnm_bad, c="grey", alpha=.2, ls="--")
    ax_i.plot(alphaRange_badKorokhin, korokhinXXnm_bad, c="grey", alpha=.2, ls="--")

    # plot the 30° and 100° intersection, and annotate the lambda eff
    ax_i.axvline(30, ls="--", c="k", alpha=.7)
    ax_i.axvline(100, ls="--", c="k", alpha=.7)
    ax_i.annotate(f"$\\lambda = {lamba}$ nm", loc_i)

    # min and max
    min_buratti, max_buratti = np.min(buratti_XXnm[good_buratti]), np.max(buratti_XXnm[good_buratti])
    min_hicks, max_hicks = np.min(hicks_XXnm[good_hicks]), np.max(hicks_XXnm[good_hicks])
    min_hillier, max_hillier = np.min(hillier_XXnm[good_hillier]), np.max(hillier_XXnm[good_hillier])
    min_korokhin, max_korokhin = np.min(korokhin_XXnm[good_korokhin]), np.max(korokhin_XXnm[good_korokhin])
    min_arr = np.asarray([min_buratti, min_hicks, min_hillier, min_korokhin])
    max_arr = np.asarray([max_buratti, max_hicks, max_hillier, max_korokhin])
    ax_i.set_ylim(np.min(min_arr[~np.isnan(min_arr)]), np.max(max_arr[~np.isnan(max_arr)]))

    # miscellaneous details
    if lamba == lamba_eff[1]:
        ax_i.legend()
    if lamba == lamba_eff[0] or lamba == lamba_eff[3]:
        ax_i.set_ylabel(r'$A_{eq}(\alpha)$', fontsize=14)
    if lamba == lamba_eff[4]:
        ax_i.set_xlabel(r'$\alpha$ (°)', fontsize=17)

fig_alpha.show()

# slope computing
lambaVelik_range = [357.9, 1_026.1]


def slope_computing(wv, al, al_rel, lamba_norm, dlamba, lims_data, alpha, alpha_rel, plot=False, title=None):

    def normalizeXX(x, y, lims):
        mask = ~np.logical_and(x > lims[0], x < lims[1])
        y_masked = np.ma.masked_array(y, mask=mask)
        median_to_norm = np.ma.median(y_masked)
        y_normXXnm = y / median_to_norm
        return y_normXXnm, median_to_norm

    alRel_norm, alRel_coeff = normalizeXX(wv, al_rel, [lamba_norm - dlamba, lamba_norm + dlamba])
    al_norm, al_coeff = normalizeXX(wv, al, [lamba_norm - dlamba, lamba_norm + dlamba])

    # mask
    mask_range = ~np.logical_and(wv > lambaVelik_range[0], wv < lambaVelik_range[1])
    wv_arr_masked = np.ma.masked_array(wv, mask=mask_range)
    alRel_norm_masked = np.ma.masked_array(alRel_norm, mask=mask_range)
    al_norm_masked = np.ma.masked_array(al_norm, mask=mask_range)
    # bad ones
    wv_bad = np.ma.masked_array(wv, mask=~mask_range)
    alRel_norm_masked_bad = np.ma.masked_array(alRel_norm, mask=~mask_range)
    al_norm_masked_bad = np.ma.masked_array(al_norm, mask=~mask_range)

    # straight line
    al_ratio_masked = al_norm_masked / alRel_norm_masked
    m, n = np.ma.polyfit(wv_arr_masked, al_ratio_masked, 1)
    if plot:
        fig_ex, ax_ex = plt.subplots()
        ax_ex.plot(wv_arr_masked, alRel_norm_masked, c="k", ls="--", label=f"$\\alpha = {alpha_rel}°$")
        ax_ex.plot(wv_bad, alRel_norm_masked_bad, c="grey", alpha=.2)
        ax_ex.plot(wv_arr_masked, al_norm_masked, c="k", label=f"$\\alpha = {alpha}°$")
        ax_ex.plot(wv_bad, al_norm_masked_bad, c="grey", alpha=.2)
        ax_ex.plot(wv_arr_masked, al_ratio_masked)
        ax_ex.plot(wv_arr_masked, wv_arr_masked * m + n, c="r")
        ax_ex.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
        ax_ex.set_ylabel(r'$A_{eq}(\lambda)$ normalized at $603$ nm', fontsize=14)
        ax_ex.legend()
        ax_ex.grid()
        if title is not None:
            ax_ex.set_title(title, fontsize=18)
        fig_ex.show()
    return m


alpha_rel_ex = 6
alpha_ex = 20
dlamba_slope = 5
lambaToNormalize = 603
# Buratti
buratti6Deg = buratti2011(lamba_buratti_toEval, alpha_rel_ex, mare=True)
burattiAlphaDeg = buratti2011(lamba_buratti_toEval, alpha_ex, mare=True)
slope_computing(lamba_buratti_toEval, burattiAlphaDeg, buratti6Deg, lambaToNormalize, dlamba_slope, lambaVelik_range,
                alpha_ex, alpha_rel_ex, plot=True, title="Buratti 2011")
# Hicks
hicks6Deg = hicks2011(lamba_hicks_toEval, alpha_rel_ex, mare=True)
hicksAlphaDeg = hicks2011(lamba_hicks_toEval, alpha_ex, mare=True)
slope_computing(lamba_hicks_toEval, hicksAlphaDeg, hicks6Deg, lambaToNormalize, dlamba_slope, lambaVelik_range,
                alpha_ex, alpha_rel_ex, plot=True, title="Hicks 2011")
# Hillier
hillier6Deg = hillier1999(lamba_hillier_toEval, alpha_rel_ex, mare=True)
hillierAlphaDeg = hillier1999(lamba_hillier_toEval, alpha_ex, mare=True)
slope_computing(lamba_hillier_toEval, hillierAlphaDeg, hillier6Deg, lambaToNormalize, dlamba_slope, lambaVelik_range,
                alpha_ex, alpha_rel_ex, plot=True, title="Hillier 2011")

# Korokhin
korokhin6Deg = korokhin2007(lamba_korokhin_toEval, alpha_rel_ex)
korokhinAlphaDeg = korokhin2007(lamba_korokhin_toEval, alpha_ex)
slope_computing(lamba_korokhin_toEval, korokhinAlphaDeg, korokhin6Deg, lambaToNormalize, dlamba_slope, lambaVelik_range,
                alpha_ex, alpha_rel_ex, plot=True, title="Korokhin 2007")

fig_slope, ax_slope = plt.subplots(1, 1, layout="constrained")
# Buratti
buratti_eqAlalphaDeg = [buratti2011(lamba_buratti_toEval, alpha_i, mare=True) for alpha_i in phase_angle_range]
buratti_slope = np.asarray([slope_computing(lamba_buratti_toEval, bu2011, buratti6Deg, lambaToNormalize,
                                            dlamba_slope, lambaVelik_range, alpha_i, alpha_rel_ex) for
                            (bu2011, alpha_i) in zip(buratti_eqAlalphaDeg, phase_angle_range)])
# Hicks
hicks_eqAlalphaDeg = [hicks2011(lamba_hicks_toEval, alpha_i, mare=True) for alpha_i in phase_angle_range]
hicks_slope = np.asarray([slope_computing(lamba_hicks_toEval, hic2011, hicks6Deg, lambaToNormalize,
                                          dlamba_slope, lambaVelik_range, alpha_i, alpha_rel_ex) for
                          (hic2011, alpha_i) in zip(hicks_eqAlalphaDeg, phase_angle_range)])
# Hillier
hillier_eqAlalphaDeg = [hillier1999(lamba_hillier_toEval, alpha_i, mare=True) for alpha_i in phase_angle_range]
hillier_slope = np.asarray([slope_computing(lamba_hicks_toEval, hill1999, hillier6Deg, lambaToNormalize,
                                            dlamba_slope, lambaVelik_range, alpha_i, alpha_rel_ex) for
                           (hill1999, alpha_i) in zip(hillier_eqAlalphaDeg, phase_angle_range)])
# Korokhin
korokhin_eqAlalphaDeg = [korokhin2007(lamba_korokhin_toEval, alpha_i) for alpha_i in phase_angle_range]
korokhin_slope = np.asarray([slope_computing(lamba_korokhin_toEval, kor2007, korokhin6Deg, lambaToNormalize,
                                             dlamba_slope, lambaVelik_range, alpha_i, alpha_rel_ex) for
                            (kor2007, alpha_i) in zip(korokhin_eqAlalphaDeg, phase_angle_range)])
# plot slopes
ax_slope.plot(phase_angle_range[good_buratti], buratti_slope[good_buratti], label="Buratti 2011")
ax_slope.plot(phase_angle_range[good_hicks], hicks_slope[good_hicks], label="Hicks 2011")
ax_slope.plot(phase_angle_range[good_hillier], hillier_slope[good_hillier], label="Hillier 1999")
ax_slope.plot(phase_angle_range[good_korokhin], korokhin_slope[good_korokhin], label="Korokhin 2007")
# bad ones
buratti_slope_bad = np.ma.masked_array(buratti_slope, mask=good_buratti)
hicks_slope_bad = np.ma.masked_array(hicks_slope, mask=good_hicks)
hillier_slope_bad = np.ma.masked_array(hillier_slope, mask=good_hillier)
korokhin_slope_bad = np.ma.masked_array(korokhin_slope, mask=good_korokhin)

# plot bad ones
ax_slope.plot(alphaRange_badBuratti, buratti_slope_bad, c="grey", alpha=.2, ls="--")
ax_slope.plot(alphaRange_badHicks, hicks_slope_bad, c="grey", alpha=.2, ls="--")
ax_slope.plot(alphaRange_badHillier, hillier_slope_bad, c="grey", alpha=.2, ls="--")
ax_slope.plot(alphaRange_badKorokhin, korokhin_slope_bad, c="grey", alpha=.2, ls="--")

# lims
min_buratti_slope, max_buratti_slope = np.min(buratti_slope[good_buratti]), np.max(buratti_slope[good_buratti])
min_hicks_slope, max_hicks_slope = np.min(hicks_slope[good_hicks]), np.max(hicks_slope[good_hicks])
min_hillier_slope, max_hillier_slope = np.min(hillier_slope[good_hillier]), np.max(hillier_slope[good_hillier])
min_korokhin_slope, max_korokhin_slope = np.min(korokhin_slope[good_korokhin]), np.max(korokhin_slope[good_korokhin])
min_slope = np.min([min_buratti_slope, min_hicks_slope, min_hillier_slope, min_korokhin_slope])
max_slope = np.max([max_buratti_slope, max_hicks_slope, max_hillier_slope, max_korokhin_slope])

ax_slope.axvline(55, c="k", ls="--", alpha=.7)
ax_slope.set_ylim(min_slope, max_slope)
ax_slope.set_xlabel(r'$\alpha$ (°)', fontsize=17)
ax_slope.set_ylabel(f"Normalized spectrum slope within ({lambaVelik_range[0]}, {lambaVelik_range[1]}) nm")
ax_slope.grid()
ax_slope.legend()
ax_slope.set_yscale("symlog", linthresh=1e-4)
fig_slope.show()

# compare albedos
alpha30_deg = 30
burattiAlpha30 = buratti2011(lamba_buratti_toEval, alpha30_deg, mare=True)
hicksAlpha30 = hicks2011(lamba_hicks_toEval, alpha30_deg, mare=True)
hillierAlpha30 = hillier1999(lamba_hillier_toEval, alpha30_deg, mare=True)
korokhinAlpha30 = korokhin2007(lamba_korokhin_toEval, alpha30_deg)

# mask
maskCompareBur = np.logical_and(lamba_buratti_toEval > lambaVelik_range[0], lamba_buratti_toEval < lambaVelik_range[1])
maskCompareHic = np.logical_and(lamba_hicks_toEval > lambaVelik_range[0], lamba_hicks_toEval < lambaVelik_range[1])
maskCompareHil = np.logical_and(lamba_hillier_toEval > lambaVelik_range[0], lamba_hillier_toEval < lambaVelik_range[1])
maskCompareKo = np.logical_and(lamba_korokhin_toEval > lambaVelik_range[0], lamba_korokhin_toEval < lambaVelik_range[1])

fig_compare, ax_compare = plt.subplots(1, 1, layout="constrained")

# Velikodsky interpolation
maskVelikData = np.logical_and(velik_table[:, 0] > lambaVelik_range[0], velik_table[:, 0] < lambaVelik_range[1])
lamba_velik_toEval = np.linspace(velik_table[:, 0][0], velik_table[:, 0][-1], 1_000)
maskCompareVelik = np.logical_and(lamba_velik_toEval > lambaVelik_range[0], lamba_velik_toEval < lambaVelik_range[1])
velik6deg = velikodsky2011(lamba_velik_toEval, 6, terrain="mare", method_interp="univariate", s=0.5,)
                           # mask_fmodel=maskVelikData)
velik100deg = velikodsky2011(lamba_velik_toEval, 100, terrain="mare", method_interp="univariate", s=0.5,)
                             # mask_fmodel=maskVelikData)


ax_compare.plot(lamba_velik_toEval[maskCompareVelik], velik6deg[maskCompareVelik],
                c="cyan", alpha=.5)
ax_compare.plot(velik_table[:, 0][maskVelikData], velik_table[:, 2][maskVelikData],
                label=r'Velikodsky 2011, $\alpha = 6°$', ls="--", c="k")
ax_compare.plot(lamba_buratti_toEval[maskCompareBur], burattiAlpha30[maskCompareBur],
                label=r'Buratti 2011, $\alpha = 30°$')
ax_compare.plot(lamba_hicks_toEval[maskCompareHic], hicksAlpha30[maskCompareHic],
                label=r'Hicks 2011, $\alpha = 30°$')
ax_compare.plot(lamba_hillier_toEval[maskCompareHil], hillierAlpha30[maskCompareHil],
                label=r'Hillier 1999, $\alpha = 30°$')
#ax_compare.plot(lamba_korokhin_toEval[maskCompareKo], korokhinAlpha30[maskCompareKo],
                #label=r'Korokhin 2007, $\alpha = 30°$')
ax_compare.plot(lamba_velik_toEval[maskCompareVelik], velik100deg[maskCompareVelik],
                c="cyan", alpha=.5)
ax_compare.plot(velik_table[:, 0][maskVelikData], eqAl_recovered[maskVelikData],
                label=r'Velikodsky 2011, $\alpha = 100°$', c="k")
ax_compare.grid()
ax_compare.legend()
ax_compare.set_ylabel(r'$A_{eq}(\lambda)$', fontsize=15)
ax_compare.set_xlabel(r'$\lambda$ (nm)', fontsize=15)
fig_compare.show()

# fig_ratio, ax_ratio = plt.subplots()
# ax_ratio.plot()





