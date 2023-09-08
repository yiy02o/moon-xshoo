from moon_preMolecfit import *
import shutil


def copy_and_update_fits(directory, ARM, normalize=False, update=False):
    for folder in os.listdir(directory):
        if folder == 'README':
            pass
        else:
            for file in os.listdir(directory + '/' + folder):
                if 'SCI_SLIT_FLUX_MERGE1D_' + ARM in file:
                    # with fits.open(directory + '/' + folder + '/' + file) as hdu1D_list:
                        # hdu1D_list.writeto(directory + '/' + folder + '/' + 'COPY_' + file)
                    src = directory + '/' + folder + '/' + file
                    dst = directory + '/' + folder + '/' + 'COPY_' + file
                    shutil.copy(src, dst)
                    break

            if update:
                upd_str = 'UPDATED_NORM_' if normalize else 'UPDATED_'
                for file in os.listdir(directory + '/' + folder):
                    if 'SCI_SLIT_FLUX_MERGE2D_' + ARM in file:
                        with fits.open(directory + '/' + folder + '/' + file) as hdu2D_list:
                            flux_2D = np.median(hdu2D_list[0].data, axis=0)
                            std = np.std(hdu2D_list[0].data, axis=0)
                            if normalize:
                                idx_min, idx_max = -10_000, -8_000
                                flux_test = flux_2D[idx_min:idx_max + 1]
                                median_test_flux = np.median(flux_test)
                                flux_2D = flux_2D/median_test_flux
                        break

                for file in os.listdir(directory + '/' + folder):
                    if 'COPY_' in file and ARM in file:
                        with fits.open(directory + '/' + folder + '/' + file, mode='update') as hdu_list:
                            print(flux_2D)
                            hdu_list[0].data = flux_2D
                            hdu_list[1].data = std
                        new_file_name = file.replace('COPY_', upd_str)
                        old_dir = directory + '/' + folder + '/' + file
                        new_dir = directory + '/' + folder + '/' + new_file_name
                        os.rename(old_dir, new_dir)
                        break

            pass
    pass


# share_path = '/home/yiyo/reflexData_mapMode_Highlands/reflex_end_products/'
#
# for direc in os.listdir(share_path):
#     copy_and_update_fits(share_path + direc + '/', 'VIS', normalize=False, update=True)
#     #print(direc)

'''
share_path = '/home/yiyo/reflex_data_telluric/reflex_end_products/2023-07-10T18:14:10/'

VIS1_tel_path = share_path + 'XSHOO.2018-03-10T10:02:53.034_tpl/Hip098197_TELL_SLIT_FLUX_MERGE1D_VIS.fits'
VIS1_2D_tel_path = share_path + 'XSHOO.2018-03-10T10:02:53.034_tpl/Hip098197_TELL_SLIT_FLUX_MERGE2D_VIS.fits'

VIS2_tel_path = share_path + 'XSHOO.2018-03-10T10:05:34.899_tpl/Hip098197_TELL_SLIT_FLUX_MERGE1D_VIS.fits'
NIR1_tel_path = share_path + 'XSHOO.2018-03-10T10:02:56.137_tpl/Hip098197_TELL_SLIT_FLUX_MERGE1D_NIR.fits'
NIR2_tel_path = share_path + 'XSHOO.2018-03-10T10:05:38.008_tpl/Hip098197_TELL_SLIT_FLUX_MERGE1D_NIR.fits'

VIS1D_path = '/home/yiyo/reflexData_mapMode_Highlands/reflex_end_products/2023-05-28T23:33:43/' \
             'XSHOO.2018-03-10T08:00:28.424_tpl/MOV_Luna_highlands_2_SCI_SLIT_FLUX_MERGE1D_VIS.fits'
VIS2D_path = '/home/yiyo/reflexData_mapMode_Highlands/reflex_end_products/2023-05-28T23:33:43/' \
             'XSHOO.2018-03-10T08:00:28.424_tpl/MOV_Luna_highlands_2_SCI_SLIT_FLUX_MERGE2D_VIS.fits'

VIS1_afterMolec_path = '/home/yiyo/reflex_data_after_molecfit/reflex_end_products/molecfit/' + \
                       'XSHOOTER/2023-07-09T02:50:21/MOV_Luna_highlands_2_SCI_SLIT_FLUX_MERGE1D_VIS/' + \
                       'MOV_Luna_highlands_2_SCIENCE_TELLURIC_CORR.fits'
VIS1_afterMolec_tel_path = '/home/yiyo/reflex_data_std_starCorr/reflex_end_products/molecfit/XSHOOTER/' \
                           '2023-07-11T07:39:08/MOV_Luna_highlands_2_SCI_SLIT_FLUX_MERGE1D_VIS/' \
                           'MOV_Luna_highlands_2_SCIENCE_TELLURIC_CORR.fits'
VIS1_tel = fits.open(VIS1_tel_path)
VIS2_tel = fits.open(VIS2_tel_path)
VIS1_2D_tel = fits.open(VIS1_2D_tel_path)
VIS1D = fits.open(VIS1D_path)
VIS2D = fits.open(VIS2D_path)
VIS1_afterMolec = fits.open(VIS1_afterMolec_path)
VIS1_afterMolec_tel = fits.open(VIS1_afterMolec_tel_path)

#VIS2 = fits.open(VIS2_path)
#NIR1 = fits.open(NIR1_path)
#NIR2 = fits.open(NIR2_path)
#VIS1_afterMolec = fits.open(VIS1_afterMolec_path)

plt.plot(VIS1_tel[0].data[700:], label='VIS telluric standard star observed at {}'.format(VIS1_tel[0].header['DATE-OBS']))
plt.plot(VIS2_tel[0].data[700:], label='VIS telluric standard star observed at {}'.format(VIS2_tel[0].header['DATE-OBS']))
#plt.plot(VIS1_tel[0].data[700:], label='VIS telluric standard star')
# plt.plot(VIS1D[0].data[700:], label='VIS 1D merge spectrum of Moon')
#plt.plot(np.median(VIS2D[0].data, axis=0)[700:], label='Median VIS 2D merge spectrum of Moon')
#plt.plot(VIS1_afterMolec[0].data[700:], label='VIS 1D merge spectrum after molecfit (original result)')
plt.ylabel('Flux')
#plt.plot(VIS1_afterMolec_tel[0].data[700:], label='VIS 1D merge spectrum after molecfit with std telluric star')
# plt.plot(np.median(VIS2[0].data, axis=0)[700:], label='VIS2')
# plt.plot(NIR1[0].data[700:], label='1')
# plt.plot(NIR2[0].data[700:], label='2')
plt.legend(fontsize=7)
plt.grid()
plt.show()

'''


