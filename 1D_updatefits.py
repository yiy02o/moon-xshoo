from moon_preMolecfit import *


def copy_and_update_fits(directory, ARM):
    for folder in os.listdir(directory):
        if folder == 'README':
            pass
        else:
            for file in os.listdir(directory + '/' + folder):
                if 'SCI_SLIT_FLUX_MERGE1D_' + ARM in file:
                    # with fits.open(directory + '/' + folder + '/' + file) as hdu_list:
                    print('aqui')
                    # hdu_list.writeto('updated_' + file)

            pass
    pass


share_path = '/home/yiyo/reflexData_mapMode_Highlands/reflex_end_products/2023-05-28T23:33:43/' \
             'XSHOO.2018-03-10T08:00:28.424_tpl/'
path_old_1D = share_path + 'MOV_Luna_highlands_2_SCI_SLIT_FLUX_MERGE1D_VIS.fits'
path_new_1D = share_path + 'MOV_Luna_highlands_2_SCI_SLIT_FLUX_MERGE1D_VIS_UPDATE.fits'
path_2D = share_path + 'MOV_Luna_highlands_2_SCI_SLIT_FLUX_MERGE2D_VIS.fits'

hdul_old = fits.open(path_old_1D)
hdul_new = fits.open(path_new_1D)
hdul_2D = fits.open(path_2D)

median_2D = np.median(hdul_2D[0].data, axis=0)
std_2D = np.std(hdul_2D[0].data, axis=0)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Antes del cambio', fontsize=24)

axs[0].plot(hdul_old[0].data[1000:], label='Old 1D')
axs[0].plot(hdul_new[0].data[1000:], label='New 1D')
axs[0].legend()
axs[0].grid()

axs[1].plot(median_2D[1000:], label='Median 2D')
axs[1].plot(hdul_new[0].data[1000:], label='New 1D')
axs[1].legend()
axs[1].grid()

plt.show()
plt.clf()


with fits.open(path_new_1D, mode='update') as fits_file:
    fits_file[0].data = median_2D
    fits_file[1].data = std_2D

fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
fig2.suptitle('Después del cambio', fontsize=24)

axs2[0].plot(hdul_old[0].data[1000:], label='Old 1D')
axs2[0].plot(hdul_new[0].data[1000:], label='New 1D')
axs2[0].legend()
axs2[0].grid()

axs2[1].plot(median_2D[1000:], label='Median 2D')
axs2[1].plot(hdul_new[0].data[1000:], label='New 1D')
axs2[1].legend()
axs2[1].grid()

plt.show()
