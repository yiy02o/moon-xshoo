from check_broadeningMoonspectra import *


def plot_resolution_intervals(figure, tw, tf_smooth_norm, tf_norm, alb, x_lims_arr, dlamba_fit):
    if not isinstance(dlamba_fit, list):
        dlamba_fit = np.repeat(dlamba_fit, len(x_lims_arr))

    def gaussian_fit(params, lamba_value):
        a, mu, sigma = params
        output = 1 + a * np.exp(-(lamba_value - mu)**2 / (2*sigma**2))
        return output

    def min_function(lamba_value, a, mu, sigma):
        params = a, mu, sigma
        return gaussian_fit(params, lamba_value)

    R_pre = []
    R_post = []
    w_offset = []
    x_lim = np.max(dlamba_fit)
    for ax_lims, x_lims, dlamba in zip(figure.axes, x_lims_arr, dlamba_fit):
        # local mask
        mask_to_plot = np.ma.where(np.logical_and(tw > x_lims[0], tw < x_lims[1]))
        tw_to_plot, tf_norm_to_plot = tw[mask_to_plot[0]], tf_norm[mask_to_plot[0]]
        tf_smooth_norm_to_plot, alb_to_plot = tf_smooth_norm[-1].data[mask_to_plot[0]], alb[-1].data[mask_to_plot[0]]
        # min mask
        mask_min = np.ma.where(np.fabs(tw_to_plot[np.argmin(tf_smooth_norm_to_plot)] - tw_to_plot) <= dlamba)
        wv_min = tw_to_plot[mask_min]
        tf_smooth_norm_min = tf_smooth_norm_to_plot[mask_min]
        tf_norm_min = tf_norm_to_plot[mask_min]
        # fit
        # smooth
        p_initial = -0.4, wv_min[np.argmin(tf_smooth_norm_min)], 0.2
        p_op, cova = curve_fit(min_function, wv_min, tf_smooth_norm_min, p0=p_initial,
                               bounds=((-1, 0, 0), (0, np.inf, np.inf)))
        ax_lims.plot(wv_min - p_op[1], tf_smooth_norm_min, c='b')
        ax_lims.plot(wv_min[np.argmin(tf_smooth_norm_min)] - p_op[1], np.min(tf_smooth_norm_min), ".", c="r")
        ax_lims.plot(wv_min - p_op[1], gaussian_fit(p_op, wv_min), color="r", ls="--",
                     label=f"Post: $R\\sim{p_op[1] / (2 * np.sqrt(2 * np.log(2)) * p_op[-1]):.0f}$")
        R_post.append(p_op[1] / (2 * np.sqrt(2 * np.log(2)) * p_op[-1]))
        w_offset.append(p_op[1])
        # tf
        p0_tf = -0.4, wv_min[np.argmin(tf_norm_min)], 0.2
        p_opt_tf, co_tf_ = curve_fit(min_function, wv_min, tf_norm_min, p0=p0_tf,
                                     bounds=((-1, 0, 0), (0, np.inf, np.inf)))
        ax_lims.plot(wv_min - p_opt_tf[1], tf_norm_min, c='k')
        ax_lims.plot(wv_min - p_opt_tf[1], gaussian_fit(p_opt_tf, wv_min), color="g", ls="--",
                     label=f"Pre: $R\\sim{p_opt_tf[1] / (2 * np.sqrt(2 * np.log(2)) * p_opt_tf[-1]):.0f}$")
        R_pre.append(p_opt_tf[1] / (2 * np.sqrt(2 * np.log(2)) * p_opt_tf[-1]))

        ax_lims.grid()
        ax_lims.legend(fontsize=6)
        ax_lims.set_ylim(np.ma.median(tf_norm) - 3*np.ma.std(tf_norm), 1.05)
        ax_lims.set_xlim(-x_lim, x_lim)
    return R_pre, R_post, w_offset


uvb_lims_tofit = [[416, 418], [468, 469.5], [495, 497], [506, 507.5], [520, 522], [522, 524], [532, 532.7], [542, 544]]
dlamba_uvb = [.2, .2, .2, .2, .2, .2, .2, .2]
fig_zoom_uvb, ax_zoom_uvb = plt.subplots(2, 4, figsize=(8, 4), layout="constrained")
R_pre_uvb, R_post_uvb, w_offset_uvb = plot_resolution_intervals(fig_zoom_uvb, wv_global, tf_broad_global,
                                                                tf_norm_global, alb_broad_global, uvb_lims_tofit,
                                                                dlamba_uvb)
fig_zoom_uvb.show()

vis_lims_tofit = [[585, 586.2], [594, 596], [603.5, 605.1], [676, 677.5], [711.5, 713],
                  [749, 750], [777, 779], [799, 800], [846, 848], [864.5, 865.5], [888, 890]]
dlamba_vis = [.4, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2, .2]
fig_zoom_vis, ax_zoom_vis = plt.subplots(2, 6, figsize=(12, 4), layout="constrained")
R_pre_vis, R_post_vis, w_offset_vis = plot_resolution_intervals(fig_zoom_vis, wv_global, tf_broad_global,
                                                                tf_norm_global, alb_broad_global, vis_lims_tofit, .3)
fig_zoom_vis.show()

nir_lims_tofit = [[1055, 1065], [1560, 1565], [1705, 1715]]
fig_zoom_nir, ax_zoom_nir = plt.subplots(3, 1, figsize=(3, 9))
R_pre_nir, R_post_nir, w_offset_nir = plot_resolution_intervals(fig_zoom_nir, wv_global, tf_broad_global,
                                                                tf_norm_global, alb_broad_global, nir_lims_tofit, 0.4)
fig_zoom_nir.show()

R_pre_global = [R_pre_uvb, R_pre_vis, R_pre_nir]
R_post_global = [R_post_uvb, R_post_vis, R_post_nir]
w_offset_global = [w_offset_uvb, w_offset_vis, w_offset_nir]
R_pre_global = np.asarray(list(chain.from_iterable(R_pre_global)))
R_post_global = np.asarray(list(chain.from_iterable(R_post_global)))
w_offset_global = np.asarray(list(chain.from_iterable(w_offset_global)))


fig_res, ax_res = plt.subplots()
ax_res.scatter(w_offset_global, R_pre_global, s=12, c="g", marker='o')
ax_res.scatter(w_offset_global, R_post_global, s=12, c="r", marker='*')
for x, y1, y2 in zip(w_offset_global, R_pre_global, R_post_global):
    ax_res.arrow(x, y1, 0, y2 - y1, width=3, length_includes_head=True, head_width=15, head_length=500,
                 color='k', alpha=.3)
ax_res.fill_between(np.linspace(300, 3_000, 20_000), 0, 20_000,
                    where=np.logical_and(np.linspace(300, 3_000, 20_000) > 320,
                                         np.linspace(300, 3_000, 20_000) < 545),
                    color="b", alpha=.25)
ax_res.fill_between(np.linspace(300, 3_000, 20_000), 0, 20_000,
                    where=np.logical_and(np.linspace(300, 3_000, 20_000) > 550,
                                         np.linspace(300, 3_000, 20_000) < 1_020),
                    color="y", alpha=.25)
ax_res.fill_between(np.linspace(300, 3_000, 20_000), 0, 20_000,
                    where=np.logical_and(np.linspace(300, 3_000, 20_000) > 1_050,
                                         np.linspace(300, 3_000, 20_000) < 3000),
                    color="r", alpha=.25)
ax_res.set_xlabel(f"$\\lambda$ (nm)")
ax_res.set_ylabel("$R$")
ax_res.grid()
ax_res.set_ylim(750, 18_500)
ax_res.set_xlim(345, 1_785)
fig_res.show()