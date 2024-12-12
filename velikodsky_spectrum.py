import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u


velik_table = np.asarray([[350, 0.0861, 0.0551, 0.0728],
                          [373, 0.0955, 0.0602, 0.0803],
                          [397, 0.1043, 0.0648, 0.0873],
                          [420, 0.1114, 0.0688, 0.0931],
                          [443, 0.1185, 0.0728, 0.0988],
                          [467, 0.1260, 0.0770, 0.1049],
                          [490, 0.1329, 0.0811, 0.1106],
                          [513, 0.1397, 0.0852, 0.1162],
                          [537, 0.1465, 0.0894, 0.1219],
                          [560, 0.1531, 0.0936, 0.1275],
                          [583, 0.1586, 0.0971, 0.1322],
                          [607, 0.1638, 0.1005, 0.1366],
                          [630, 0.1692, 0.1038, 0.1411],
                          [653, 0.1741, 0.1070, 0.1452],
                          [677, 0.1788, 0.1100, 0.1492],
                          [700, 0.1834, 0.1130, 0.1531],
                          [723, 0.1878, 0.1160, 0.1569],
                          [747, 0.1917, 0.1184, 0.1602],
                          [770, 0.1950, 0.1205, 0.1629],
                          [793, 0.1972, 0.1218, 0.1648],
                          [800, 0.1981, 0.1234, 0.1660],
                          [850, 0.2018, 0.1250, 0.1688],
                          [900, 0.2060, 0.1265, 0.1718],
                          [950, 0.2121, 0.1290, 0.1764],
                          [1_000, 0.2200, 0.1330, 0.1826],
                          [1_050, 0.2290, 0.1384, 0.1900],
                          [1_100, 0.2381, 0.1451, 0.1981],
                          [1_150, 0.2468, 0.1519, 0.2060],
                          [1_200, 0.2551, 0.1585, 0.2136],
                          [1_250, 0.2632, 0.1649, 0.2209],
                          [1_300, 0.2702, 0.1704, 0.2273],
                          [1_350, 0.2763, 0.1748, 0.2326],
                          [1_400, 0.2837, 0.1797, 0.2390],
                          [1_450, 0.2921, 0.1857, 0.2463],
                          [1_500, 0.2985, 0.1909, 0.2523],
                          [1_550, 0.3037, 0.1952, 0.2571],
                          [1_600, 0.3084, 0.1990, 0.2613],
                          [1_650, 0.3126, 0.2024, 0.2652],
                          [1_700, 0.3163, 0.2052, 0.2686],
                          [1_750, 0.3197, 0.2077, 0.2715],
                          [1_800, 0.3232, 0.2100, 0.2745],
                          [1_850, 0.3271, 0.2122, 0.2777],
                          [1_900, 0.3310, 0.2146, 0.2810],
                          [1_950, 0.3350, 0.2174, 0.2844],
                          [2_000, 0.3391, 0.2203, 0.2880],
                          [2_050, 0.3438, 0.2233, 0.2920],
                          [2_100, 0.3491, 0.2267, 0.2964],
                          [2_150, 0.3548, 0.2304, 0.3013],
                          [2_200, 0.3610, 0.2349, 0.3068],
                          [2_250, 0.3677, 0.2400, 0.3128],
                          [2_300, 0.3747, 0.2457, 0.3192],
                          [2_350, 0.3823, 0.2525, 0.3265],
                          [2_400, 0.3902, 0.2601, 0.3342],
                          [2_450, 0.3973, 0.2675, 0.3415],
                          [2_500, 0.4022, 0.2730, 0.3466]])


def recover_Aeq(alpha_deg, plot=False, terrain="mare"):
    if terrain != "mare" and terrain != "highland" and terrain != "average":
        raise ValueError("Try again with another terrain.")
    alpha_rad = alpha_deg * u.deg.to(u.rad)
    velik_aEq6 = velik_table[:, 1] if terrain == "highland" else velik_table[:, 2] if terrain == "mare" else \
        velik_table[:, 3]

    def normalizeXX(wv, al, norm_range):
        mask_range = ~np.logical_and(wv > norm_range[0], wv < norm_range[1])
        y_arr_masked = np.ma.masked_array(al, mask=mask_range)
        median_to_norm = np.ma.median(y_arr_masked)
        y_normXXnm = al / median_to_norm
        return y_normXXnm, median_to_norm

    x = velik_table[:, 0]

    y1, norm_coeff = normalizeXX(x, velik_aEq6, [590, 610])
    delta_m = 6e-5 * (u.AA.to(u.nm)) ** -1
    delta_n = 1 - delta_m * 603
    y2 = y1 * (delta_m * x + delta_n)
    eqAl6alpha = velik_table[:, 2]
    # coeff 603 nm
    A2, mu2, A3, mu3, A4 = .042, 8.6, .074, .777, .0065
    eqAl630nm_alpha = A2 * np.exp(-mu2 * alpha_rad) + A3 * np.exp(-mu3 * alpha_rad) + A4
    eqAl55alpha = y2 * eqAl630nm_alpha
    print(norm_coeff)

    if plot:
        # plot the albedo spectrum provided in velik_table
        fig_vel, ax_vel = plt.subplots()
        ax_vel.plot(velik_table[:, 0], velik_table[:, 1], "v", c="b", label="Highland")
        ax_vel.plot(velik_table[:, 0], velik_table[:, 2], "v", c="r", label="Mare")
        ax_vel.plot(velik_table[:, 0], velik_table[:, 3], "v", c="g", label="Average Moon")
        ax_vel.set_xlabel(r'$\lambda$ (nm)', fontsize=15)
        ax_vel.set_ylabel(r'$A_{eq}(6°)$', fontsize=15)
        ax_vel.grid()
        ax_vel.legend()
        fig_vel.show()

        # plot the eq albedo normalized at 630 nm (Fig. 2  in Berezhnoi 2023)
        fig_zoom, ax_zoom = plt.subplots()
        mask_zoom = np.logical_and(x > 300, x < 800)
        ax_zoom.axhline(1, c="grey", alpha=.4)
        ax_zoom.axvline(603, c="grey", alpha=.4)
        ax_zoom.plot(x[mask_zoom], y1[mask_zoom], ls="--", c="k", label=r'$\alpha = 6°$')
        ax_zoom.plot(x[mask_zoom], y1[mask_zoom] * (delta_m * x[mask_zoom] + delta_n),
                     c="k", label=r'$\alpha > 55°$')
        ax_zoom.legend()
        ax_zoom.set_ylabel(r'$A_{eq}(\lambda)$, normalized at 603 nm', fontsize=15)
        ax_zoom.set_xlabel(r'$\lambda$ (nm)', fontsize=15)
        fig_zoom.show()

        # plot the eq albedo
        fig_eqAl, ax_eqAl = plt.subplots()
        ax_eqAl.plot(x, eqAl6alpha, ls="--", c="k", label=r'$\alpha = 6°$')
        ax_eqAl.plot(x, eqAl55alpha, c="k", label=f"$\\alpha = {alpha_deg}°$")
        ax_eqAl.axvspan(357.9, 1_026.1, color="gray", alpha=.1)
        ax_eqAl.legend()
        ax_eqAl.set_ylabel(r'$A_{eq}(\lambda)$', fontsize=15)
        ax_eqAl.set_xlabel(r'$\lambda$ (nm)', fontsize=15)
        fig_eqAl.show()
    return eqAl55alpha


eqAl_recovered = recover_Aeq(100, plot=True)
