import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow_plot(path_source, axes, loc, together=False, fontsize=10):
    ims = []
    z_lims = []
    dates = []
    for p in os.listdir(path_source):
        hdul = fits.open(path_source + p)
        print(hdul[0].header['HIERARCH ESO OBS NAME'])
        if loc in hdul[0].header['HIERARCH ESO OBS NAME'] and hdul[0].header['HIERARCH ESO SEQ OBS TYPE'] == 'OBJECT':
            arm = hdul[0].header['HIERARCH ESO SEQ ARM']
            date_obs = hdul[0].header['DATE-OBS']
            dates.append(date_obs)
            if arm == 'NIR':
                im_data = hdul[0].data
            if arm == 'VIS':
                im_data = ndimage.rotate(hdul[0].data, -90)
            if arm == 'UVB':
                im_data = ndimage.rotate(hdul[0].data, 90)
            z = ZScaleInterval()
            z1, z2 = z.get_limits(im_data)
            print('{} and {} are z1 and z2 of the dataset from {}'.format(z1, z2, date_obs))
            ims.append(im_data)
            z_lims.append([z1, z2])
            hdul.close()
    idx_order = np.argsort(dates)
    ims = np.asarray(ims)[idx_order]
    z_lims = np.asarray(z_lims)[idx_order]
    dates = np.sort(dates)
    Z_min, Z_max = np.min(np.transpose(z_lims)[0]), np.max(np.transpose(z_lims)[1])
    # print('Mínimo y máximo entre zonas son {} y {} respectivamente'.format(Zmin, Zmax))
    print(len(ims))
    for i, (ax, Z, lims, d) in enumerate(zip(fig.axes, ims, z_lims, dates)):
        print(i)
        im = ax.imshow(Z, origin='lower', cmap=plt.cm.gray, vmin=lims[0], vmax=lims[1])
        ax.text(5, 5, d, bbox={'facecolor': 'white', 'pad': 10})
        if together:
            im = ax.imshow(Z, origin='lower', cmap=plt.cm.gray, vmin=Z_min, vmax=Z_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_xticks([])
        ax.set_yticks([])
    pass


path = '/home/yiyo/Downloads/ESO_MOON_DATA_XSHOOTER/VIS/SLT_OBJ/'

#plt.clf()
#fig, axes = plt.subplots(1, 1, figsize=(20, 10), sharex='row')
#plt.rcParams['figure.constrained_layout.use'] = True


#imshow_plot(path, fig, 'maria', together=False)

#fig.suptitle('VIS Mare raw data', fontsize=30);
#plt.show()


path_true = '/home/yiyo/moon_vis_test/reflex_tmp_products/xshooter/' \
            'xsh_respon_slit_nod_1/2024-04-25T03:48:49.140/FLUX_SLIT_FLUX_MERGE1D_VIS.fits'
path_false = '/home/yiyo/moon_xshoo_pre_molecfit/reflexData_mapMode_highlands/reflex_tmp_products/xshooter/' \
             'xsh_respon_slit_nod_1/2023-05-29T00:21:32.567/FLUX_SLIT_FLUX_MERGE1D_VIS.fits'

hdul_true = fits.open(path_true)
hdul_false = fits.open(path_false)
plt.plot(hdul_true[0].data[1_000:])
plt.plot(hdul_false[0].data[1_000:])
plt.show()
