from itertools import chain
import os
import io
import numpy as np
import astropy.coordinates as apc
import astropy.time as apt
import astropy.units as u
from astropy.io import fits
#from procastro.astro import body_map
from astroquery.simbad import Simbad
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Rectangle, Circle
from matplotlib import patheffects
from PIL import Image
import pandas as pd
from moon_after_molecfit import *


def add_label(axes, moon_r, text, to_earth, azimuth, offset_label=-200, ha="left", va="center", **kwargs):
    coordinates = ([(moon_r * np.sin(to_earth) * np.sin(azimuth)).value,
                    (moon_r * np.sin(to_earth) * np.cos(azimuth)).value])
    axes.plot(*coordinates, 'o', **kwargs)
    axes.annotate(text, coordinates, (coordinates[0], coordinates[1] + offset_label), ha=ha, va=va, **kwargs)


def add_label_xy(axes, text, xyz, offset_label, **kwargs):
    axes.annotate(f"{text}{'$_{{bhd}}$' if xyz[2] < 0 else ''}", xyz[[0, 1]], (xyz[0] + offset_label[0],
                                                                               xyz[1] + offset_label[1]),
                  arrowprops={'arrowstyle': '->'}, **kwargs)


def sub_earth_n_position_angle(body, moon_object):
    elongation = moon_object.separation(body)
    to_sub_earth_angle = (np.arctan2(body.distance * np.sin(elongation),
                                     moon_object.distance - body.distance * np.cos(elongation)))
    position_angle = moon_object.position_angle(body)

    return to_sub_earth_angle, position_angle


def shadow(axes, moon_r, delta, azimuth, **kwargs):
    xy = []
    for angle in np.linspace(0, 360, 50) * u.deg:
        coo = np.array([0, np.cos(angle), np.sin(angle)]) * u.deg
        ncoo = R.from_euler("y", -delta.to(u.deg).value, degrees=True).apply(coo)
        nncoo = R.from_euler("x", -azimuth.to(u.deg).value, degrees=True).apply(ncoo)
        nncoo *= moon_r.value
        if nncoo[0] < 0:
            xy.append(nncoo[[1, 2]])
    axes.plot(*list(zip(*xy)), **kwargs)


def from_skyPosition_2_planetPosition(obs_pos, planet_pos, rotation_values, equatorial_radius, ob, n_points=100):
    """
    Sky observation ---> Planet observation
    This function transform the (RA, DEC) coordinates of an observation
    to position over a planet surface. The values returned are Longitud and Latitude
    and the (x, y, z) point over the oblate planet.

    Parameters:
    -obs_pos: A 2D-array with the Right Ascension and Declination
     coordinates of every observation of the planet. It must be in degrees. 
        For example: np.array([[284.84360, -22.54603],
                               [284.84365, -22.54603],
                               [284.84368, -22.54603],...])

    -planet_pos: A 2D-array with the Right Ascension and Declination
     coordinates of the center of the target body at every observation point. 
     It must be in degrees. 
        For example: np.array([[284.84360, -22.54603],
                               [284.84365, -22.54603],
                               [284.84368, -22.54603],...]) 
    An easy way to obtain the is through the web page "Horizons". 
    link: https://ssd.jpl.nasa.gov/horizons/app.html#/
    The R.A.___(ICRF)___DEC parameter represents it.

    -rotation_values: a list of dimensions (N, 3), where 
    N corresponds to the number of observations. 
    It contains in the first row the Longitude of target's center 
    for all the observations. In the second row are the Latitudes of target's 
    center (Planetocentric Latitude). 
    In the last row are the Angles between the body's North Pole and the 
    Earth's North Pole.
        For example:   [[328.369796, 30.067360, 40.58190],
                        [331.654933, 30.067354, 40.58190],
                        ...]]
    All these parameters can be found in Horizons. 
    The parameters are called: ObsSub_LON, obsSub_LAT, NP.ang.
    *** The Latitude given by Horizons needs to be transformed from
    Planetodetic to Planetocentric.

    -equatorial_radius: Angular radius of the target given in 
    arcsec. It can be found in Horizons as Ang-diam.
        For example: 16.96030

    -oblatitude: the ratio between the planets radius
        For example: Equatorial_radius/North_pole_radius 

    Returns:
    -A plot showing the observations' positions in the sky with 
    respect to the object center and the places in the body of 
    the same observations considering the rotation.
    -Three arrays. The first two contain the Longitude and Latitude
    of the observations over the planet. The last array has the coordinates
    (x, y, z) of the observation over the planet
    """
    
    N = len(obs_pos)
    radius = equatorial_radius / (60 * 60)
    radius_a = radius
    radius_b = radius * ob

    yy = (obs_pos[:, 0] - planet_pos[:, 0]) * np.cos(obs_pos[:, 1]*np.pi/180)
    zz = obs_pos[:, 1] - planet_pos[:, 1]
    dist = radius ** 2 - zz ** 2 - yy ** 2

    grid_long = np.linspace(-180, 180, n_points)
    grid_lat = np.linspace(-90, 90, n_points)
    grilla = np.meshgrid(grid_long, grid_lat, copy=True,
                         sparse=False, indexing='xy')

    x_coor_rf = radius_a * \
        np.cos(grilla[1] * np.pi/180) * np.cos(grilla[0] * np.pi/180)
    y_coor_rf = radius_a * \
        np.cos(grilla[1] * np.pi/180) * np.sin(grilla[0] * np.pi/180)
    z_coor_rf = radius_b * np.sin(grilla[1] * np.pi/180)

    def find_closest(y_point, z_point, x_coo, y_coo, z_coo):
        closest_dist = radius
        closest_x = 10
        for i in range(len(grid_long)):
            for j in range(len(grid_lat)):
                distance = (y_coo[i][j] - y_point) ** 2 + \
                           (z_coo[i][j] - z_point) ** 2
                if distance < closest_dist:
                    if x_coo[i][j] <= 0:
                        closest_x = np.abs(x_coo[i][j])
                        closest_dist = distance
        return closest_x

    rotated_oblato = np.zeros((N, 3, len(x_coor_rf), len(x_coor_rf[0, :])))
    for k in range(N):
        ang = rotation_values[k]
        rotation3 = Rotation.from_euler('zyx', [ang[0],
                                                -ang[1], -ang[2]], degrees=True)
        for i in range(len(grid_long)):
            for j in range(len(grid_lat)):
                oblato = np.array(
                    [x_coor_rf[i, j], y_coor_rf[i, j], z_coor_rf[i, j]])
                rotated_oblato[k, :, i, j] = rotation3.apply(oblato)

    xx = np.zeros(len(yy))
    for k in range(N):

        x_coor = rotated_oblato[k, 0, :, :]
        y_coor = rotated_oblato[k, 1, :, :]
        z_coor = rotated_oblato[k, 2, :, :]

        if ob == 1:
            xx = -np.sqrt(np.abs(dist))
        else:
            xx[k] = -find_closest(yy[k], zz[k], x_coor, y_coor, z_coor)

    obs_positions = np.zeros((N, 3))
    origin_frame = np.zeros((N, 3))

    for i in range(N):
        obs_positions[i] = np.array((xx[i], yy[i], zz[i]))
        if dist[i] < 0:
            origin_frame[i] = obs_positions[i]
        else:
            ang = rotation_values[i]

            # rotation of the target in such a way that the point (Lat=0, Lon=0) is pointing to us.
            # as well as the planet rotation axes is aligned with the earth axes.
            # LONG LAT NP
            rotation1 = Rotation.from_euler('xyz', [ang[2],
                                                    ang[1], -ang[0]], degrees=True)
            origin_frame[i] = rotation1.apply(obs_positions[i])

    rest_xx = origin_frame[:, 0]
    rest_yy = origin_frame[:, 1]
    rest_zz = origin_frame[:, 2]

    latitude = np.zeros((N))
    longitude = np.zeros((N))
    for i in range(N):
        if dist[i] < 0:
            latitude[i] = None
            longitude[i] = None
        else:
            latitude[i] = np.arcsin(rest_zz[i]/radius_b)
            longitude[i] = np.arcsin(
                rest_yy[i]/(radius_a*np.cos(latitude[i]))) * 180/np.pi
    latitude = latitude * 180/np.pi
    return longitude, latitude, origin_frame, x_coor, y_coor, z_coor


def point_of_view(coord_obs: object, rotation_values: object, equatorial_radius: object,
                  ob: object, naxis: object) -> object:
    """
    Sky observation ---> Planet observation
    This function transform the (RA, DEC) coordinates of an observation
    to position over a planet surface. The values returned are Longitude and Latitude
    and the (x, y, z) point over the oblate planet.

    Parameters:
    -coord_obs

    -rotation_values: a list of dimensions (N, 3), where 
    N corresponds to the number of observations. 
    It contains in the first row the Longitude of target's center 
    for all the observations. In the second row are the Latitudes of target's 
    center (Planetocentric Latitude). 
    In the last row are the Angles between the body's North Pole and the 
    Earth's North Pole.
        For example:   [[328.369796, 30.067360, 40.58190],
                        [331.654933, 30.067354, 40.58190],
                        ...]]
    All these parameters can be found in Horizons. 
    The parameters are called: ObsSub_LON, obsSub_LAT, NP.ang.
    *** The Latitude given by Horizons needs to be transformed from
    Planetodetic to Planetocentric.

    -equatorial_radius: Angular radius of the target given in 
    arcsec. It can be found in Horizons as Ang-diam.
        For example: 16.96030

    -oblatitude: the ration between the planets radius
        For example: Equatorial_radius/North_pole_radius 

    Returns:
    -The coordinates (x, y, z) of observation from the observer point of view.
    """

    radius = equatorial_radius / (60 * 60)
    radius_a = radius
    radius_b = radius * ob
    N = len(coord_obs)

    yy = coord_obs[:, 1]
    zz = coord_obs[:, 2]
    dist = radius ** 2 - zz ** 2 - yy ** 2

    returned_positions = np.zeros((N, 3))
    for i in range(N):
        if dist[i] < 0:
            returned_positions[i] = coord_obs[i]
        else:
            ang = rotation_values[naxis]

            # rotation of the target in such a way that the point (Lat=0, Lon=0) is pointing to us.
            # as well as the planet rotation axes is aligned with the earth axes.
            # LONG LAT NP
            rotation3 = Rotation.from_euler('zyx', [ang[0],
                                                    -ang[1], -ang[2]], degrees=True)
            returned_positions[i] = rotation3.apply(coord_obs[i])
    return returned_positions


def pos_ra_dec(path, to_csv=False):
    pos_obs = []
    moon_center = []
    date_observation = []
    for file in os.listdir(path):
        pathfile = os.path.join(path, file)
        hdul = fits.open(pathfile)
        if hdul[0].header['HIERARCH ESO DPR TYPE'] == 'SKY':  # we will include the sky files soon
            continue
        if hdul[0].header['DATE-OBS'] == '2018-03-10T08:27:22.037':#i == 18:
            continue
        ra_obs, dec_obs = hdul[0].header['RA'], hdul[0].header['DEC']
        ra_off, dec_off = hdul[0].header['HIERARCH ESO SEQ CUMOFF RA'], hdul[0].header['HIERARCH ESO SEQ CUMOFF DEC']
        pos_obs_i = [ra_obs, dec_obs]
        moon_center_i = [ra_obs - ra_off/3600, dec_obs - dec_off/3600]  # offset coordinates are in arcsec
        pos_obs.append(pos_obs_i)
        moon_center.append(moon_center_i)
        date_observation.append(hdul[0].header['DATE-OBS'])
    pos_obs = np.asarray(pos_obs)
    moon_center = np.asarray(moon_center)
    date_observation = np.asarray(date_observation)
    if to_csv:
        df = pd.DataFrame({"Date": date_observation,
                           "RA": pos_obs[:, 0],
                           "DEC": pos_obs[:, 1]})
        df.to_csv("/home/yiyo/Downloads/pos_and_dates.csv", index=False)
    return pos_obs, moon_center, date_observation


def plot_guideStars_and_moon(axes, moon_r, filename, d, start, off_arr, r_search, Gmag0, Jmag0, FOV, std_pos_arr=None,
                             guide_pos_arr=None, end=None):
    # file = np.loadtxt(filename, delimiter=",", skiprows=1, dtype='str')
    # Masking day
    # day_str = str(d) if len(str(d)) > 1 else "0" + str(d)
    # file = file[np.where(np.char.find(np.char.lower(np.core.defchararray.replace(file.T[0], '2024', '')),
    #                                  day_str) > -1)[0]]
    # Masking time --> start and end
    # file_start = file[np.where(np.char.find(np.char.lower(file.T[1]), start) > -1)[0]]
    #if not len(file_start):
    #    raise ValueError("You should change your eph log file")
    extra_loc = apc.EarthLocation.of_site("lasilla")
    moo = apc.get_body("moon", start, location=extra_loc)
    coo_moon_initial = apc.SkyCoord(moo.ra.value * u.deg, moo.dec.value * u.deg, frame='icrs')
    # print(f"Moon's inital position at {start} is ra = {coo_moon_initial.ra.value:.4f} deg, "
    #       f"dec = {coo_moon_initial.dec.value:.4f} deg")
    #if end is not None:
    #    file_end = file[np.where(np.char.find(np.char.lower(file.T[1]), end) > -1)[0]]
    #    coo_moon_final = apc.SkyCoord(file_end[0][2], file_end[0][3], frame='icrs')
        # print(f"Moon's final position at {end} is ra = {coo_moon_final.ra.value:.4f} deg, "
        #       f"dec = {coo_moon_final.dec.value:.4f} deg")
    ra_off_arr = [(ra_off * u.arcsec).to(u.deg).value + coo_moon_initial.ra.value for ra_off in off_arr.T[0]]
    dec_off_arr = [(dec_off * u.arcsec).to(u.deg).value + coo_moon_initial.dec.value for dec_off in off_arr.T[1]]
    sky_coo_reg_arr = [apc.SkyCoord(ra_off * u.deg, dec_off * u.deg,
                                    frame='icrs') for ra_off, dec_off in zip(ra_off_arr, dec_off_arr)]
    regions = ['Tycho Crater', 'Apollo 16', 'Mare Serenatis', 'West Mare Imbrium', 'Mare Humorum', 'Mare Nubium',
               'Mare Tranquillitatis']
    color_zones = ["b", "orange", "g", "k", "m", "yellow", "lime"]
    for sky_coo_reg, ra_off_reg, dec_off_reg, reg, c in zip(sky_coo_reg_arr, off_arr.T[0], off_arr.T[1], regions,
                                                            color_zones):
        axes.scatter(ra_off_reg, dec_off_reg, c=c,
                     label=reg + f": $\\alpha = {sky_coo_reg.ra.to(u.deg).value:.5f}$, "
                           f"$\\delta = {sky_coo_reg.dec.to(u.deg).value:.5f}$")

    def choose_stars(region, r, mag, date_today, target="guide"):
        Simbad.clear_cache()
        Simbad.add_votable_fields('flux(G)', 'flux(J)', 'otype', 'sp', 'pmra', 'pmdec', 'parallax')
        Simbad.TIMEOUT = 1000
        obj_table = Simbad.query_region(region, radius=r)
        if target == "guide":
            df_table = obj_table.to_pandas().dropna(subset=["FLUX_G", "FLUX_J", "SP_TYPE", "PMRA", "PMDEC",
                                                            "PLX_VALUE"])
            obj_brighter = df_table[df_table.FLUX_G < mag]
        else:
            df_table = obj_table.to_pandas().dropna(subset=["FLUX_J", "SP_TYPE", "PMRA", "PMDEC", "PLX_VALUE"])
            obj_brighter = df_table[df_table.FLUX_J < mag]
        otype = ["Star", "HighPM*", "**"]
        mask_otype = np.any([obj_brighter.OTYPE == ot for ot in otype], axis=0)
        otype_brighter = obj_brighter[mask_otype]
        otype_spt = otype_brighter.SP_TYPE.to_numpy()
        mask_spt = otype_spt != ''
        otype_id = otype_brighter.MAIN_ID.to_numpy()[mask_spt]
        otype_coord = otype_brighter.RA.to_numpy()[mask_spt] + " " + otype_brighter.DEC.to_numpy()[mask_spt]
        otype_spt = otype_spt[mask_spt]
        pm_ra_cosdec_guide = otype_brighter.PMRA.to_numpy()[mask_spt]
        pm_dec_guide = otype_brighter.PMDEC.to_numpy()[mask_spt]
        plx_guide = otype_brighter.PLX_VALUE.to_numpy()[mask_spt]
        coo_guide_otype = [apc.SkyCoord(coo, unit=(u.hourangle, u.deg), frame='icrs',
                                        pm_ra_cosdec=pm_ra_cosdec*u.mas/u.yr, pm_dec=pm_dec*u.mas/u.yr,
                                        obstime=apt.Time(2000.0, format='decimalyear'),
                                        distance=apc.Distance(parallax=plx*u.mas))
                           for coo, pm_ra_cosdec, pm_dec, plx in zip(otype_coord, pm_ra_cosdec_guide, pm_dec_guide,
                                                                     plx_guide)]
        coo_guide_otype = [coo.apply_space_motion(date_today) for coo in coo_guide_otype]
        if target == "guide":
            return coo_guide_otype, otype_id, otype_spt, otype_brighter.FLUX_J.to_numpy()[mask_spt],\
                otype_brighter.FLUX_G.to_numpy()[mask_spt]
        else:
            return coo_guide_otype, otype_id, otype_spt, otype_brighter.FLUX_J.to_numpy()[mask_spt]

    coo_guide_stars, stars_id, stars_spt, stars_Jmag, stars_Gmag = choose_stars(coo_moon_initial, r_search,
                                                                                Gmag0, start)
    off_guide_stars = []
    for coo_star in coo_guide_stars:
        first_moon_cond = np.logical_and(coo_moon_initial.ra.deg > 270., coo_moon_initial.ra.deg < 360.)
        second_moon_cond = np.logical_and(coo_moon_initial.ra.deg > 0., coo_moon_initial.ra.deg < 90.)
        first_star_cond = np.logical_and(coo_star.ra.deg > 270., coo_star.ra.deg < 360.)
        second_star_cond = np.logical_and(coo_star.ra.deg > 0., coo_star.ra.deg < 90.)
        if np.logical_and(first_moon_cond, second_star_cond):
            off_star = [(coo_star.ra.arcsec + 360*u.deg.to(u.arcsec)) - coo_moon_initial.ra.to(u.arcsec).value,
                        coo_star.dec.arcsec - coo_moon_initial.dec.to(u.arcsec).value]
        if np.logical_and(second_moon_cond, first_star_cond):
            off_star = [coo_star.ra.arcsec - (coo_moon_initial.ra.to(u.arcsec).value + 360*u.deg.to(u.arcsec)),
                        coo_star.dec.arcsec - coo_moon_initial.dec.to(u.arcsec).value]
            print(off_star[0])
            print(off_star[1])
        else:
            off_star = [coo_star.ra.arcsec - coo_moon_initial.ra.to(u.arcsec).value,
                        coo_star.dec.arcsec - coo_moon_initial.dec.to(u.arcsec).value]
        off_guide_stars.append(off_star)
    #off_guide_stars = [[coo_star.ra.arcsec - coo_moon_initial.ra.to(u.arcsec).value,
                        #coo_star.dec.arcsec - coo_moon_initial.dec.to(u.arcsec).value] for coo_star in coo_guide_stars]
    c_guide = cm.rainbow(np.linspace(0.7, 1, len(stars_id))) if guide_pos_arr is None else ['orange']
    if guide_pos_arr is not None:
        coo_guide_stars = [guide_pos_arr]
        off_guide_stars = [[guide_pos_arr[0] * u.deg.to(u.arcsec) - coo_moon_initial.ra.to(u.arcsec).value,
                            guide_pos_arr[1] * u.deg.to(u.arcsec) - coo_moon_initial.dec.to(u.arcsec).value]]
    stars = []
    count = 0
    print(stars_id)
    for i, (color, guide_name, s, j_mag, g_mag, offset) in enumerate(zip(c_guide, stars_id, stars_spt, stars_Jmag,
                                                                         stars_Gmag, off_guide_stars)):
        limb_distance = np.sqrt(offset[0] ** 2 + offset[1] ** 2) - moon_r.value
        limb_distance_criteria = np.logical_and((np.fabs(limb_distance)*u.arcsec).to(u.arcmin).value > 3.,
                                                (np.fabs(limb_distance)*u.arcsec).to(u.arcmin).value < 10.)
        if limb_distance_criteria or guide_pos_arr is not None:
            print(f"Guide star ra = {coo_guide_stars[i].ra.value if guide_pos_arr is None else coo_guide_stars[0][0]}"
                  f"deg, "f"dec = {coo_guide_stars[i].dec.value if guide_pos_arr is None else coo_guide_stars[0][1]} "
                  f"deg")
            ra_guide = coo_guide_stars[i].ra.value if guide_pos_arr is None else coo_guide_stars[0][0]
            dec_guide = coo_guide_stars[i].dec.value if guide_pos_arr is None else coo_guide_stars[0][1]
            label = f"{guide_name} {s} (guide star)  $G = {g_mag:.2f}$, $J = {j_mag:.2f}$: $\\alpha = {ra_guide:.5f}$,"\
                    f" $\\delta = {dec_guide:.5f}$" if guide_pos_arr is None else f"Guide star: " \
                                                                              f"$\\alpha = {ra_guide:.5f}$, " \
                                                                              f"$\\delta = {dec_guide:.5f}$"
            axes.plot(offset[0], offset[1], '*', label=label, c=color, markersize=12)
            coo_std_stars, std_id, std_spt, std_Jmag = choose_stars(coo_guide_stars[i] if guide_pos_arr is None else
                                                                    apc.SkyCoord(guide_pos_arr[0], guide_pos_arr[1],
                                                                                 unit=(u.deg, u.deg), frame='icrs'),
                                                                    FOV*u.arcmin, Jmag0, start, target='std')
            off_std_stars = [[coo_star.ra.arcsec - coo_moon_initial.ra.to(u.arcsec).value,
                              coo_star.dec.arcsec - coo_moon_initial.dec.to(u.arcsec).value] for coo_star in
                             coo_std_stars]
            if std_pos_arr is not None:
                off_std_stars = [[std_pos_arr[0] * u.deg.to(u.arcsec) - coo_moon_initial.ra.to(u.arcsec).value,
                                  std_pos_arr[1] * u.deg.to(u.arcsec) - coo_moon_initial.dec.to(u.arcsec).value]]
            c_std = cm.rainbow(np.linspace(0, 0.7, len(std_id))) if std_pos_arr is None and guide_pos_arr is None else \
                    cm.rainbow(np.linspace(0, 1, len(std_id))) if std_pos_arr is None and guide_pos_arr is not None \
                        else ['purple']
            for idx, (color_std, name, spt, jmag_std, off_std) in enumerate(zip(c_std, std_id, std_spt, std_Jmag,
                                                                                off_std_stars)):
                same_guide_star = np.any(np.char.find(stars_id.tolist(), name) > -1)
                repeated_std_star = np.any(np.char.find(stars, name) > -1) if count != 0 else False
                if same_guide_star or repeated_std_star:
                    continue
                count += 1
                stars.append(name)
                label_std = f"{name} ({spt}) $J = {jmag_std:.2f}$: $\\alpha = {coo_std_stars[idx].ra.value:.5f}$, " \
                            f"$\\delta = {coo_std_stars[idx].dec.value:.5f}$" if std_pos_arr is None else \
                    f"Std star: $\\alpha = {std_pos_arr[0]:.5f}$, $\\delta = {std_pos_arr[1]:.5f}$"
                # axes.plot(off_std[0], off_std[1], '*', label=label_std, c=color_std, markersize=12)
            axes.arrow(moon_r.value * np.sin(np.arctan2(offset[0], offset[1])),
                       moon_r.value * np.cos(np.arctan2(offset[0], offset[1])),
                       offset[0] - moon_r.value * np.sin(np.arctan2(offset[0], offset[1])),
                       offset[1] - moon_r.value * np.cos(np.arctan2(offset[0], offset[1])),
                       length_includes_head=True, width=10, color='k', alpha=0.6)
            axes.annotate(f"$\Delta = {(limb_distance * u.arcsec).to(u.arcmin).value:.2f} '$", (offset[0] + 100,
                                                                                                offset[1] + 100))
            field_of_view = Circle((offset[0], offset[1]), radius=FOV*u.arcmin.to(u.arcsec), edgecolor='k', fill=False)
            # field_of_view = Circle((offset[0], offset[1]), radius=108*u.arcmin.to(u.arcsec)/2, edgecolor='k',
                                   # fill=False)
            axes.add_patch(field_of_view)
            axes.set_title(f"April {d} at {start}:00UT", fontsize=35)
        print(f"Distance to the limb: {(limb_distance*u.arcsec).to(u.arcmin).value:.2f} arc minutes")
    return sky_coo_reg


def tarot_log_to_csv(filename, day_arr, init_arr, final_arr, time_span, name_to_save=None):
    file = np.loadtxt(filename, delimiter=",", skiprows=1, dtype='str')

    def search_condition(file_daily, low_lim, up_lim, delta_time):
        hour_num = [int(hour.split(":")[0]) if int(hour.split(":")[0][0]) != 0 else \
                        int(hour.split(":")[0][-1]) for hour in file_daily.T[1]]
        minute_num = [int(hour.split(":")[-1]) if int(hour.split(":")[-1][0]) != 0 else \
                          int(hour.split(":")[-1][-1]) for hour in file_daily.T[1]]
        time_num = np.array([h + m/60 for h, m in zip(hour_num, minute_num)])
        start_hour_num = int(low_lim.split(":")[0]) if int(low_lim.split(":")[0][0]) != 0 else \
            int(low_lim.split(":")[0][-1])
        start_min_num = int(low_lim.split(":")[-1])/60 if int(low_lim.split(":")[-1][0]) != 0 else \
            int(low_lim.split(":")[-1][-1])/60
        final_hour_num = int(up_lim.split(":")[0]) if int(up_lim.split(":")[0][0]) != 0 else \
            int(up_lim.split(":")[0][-1])
        final_min_num = int(up_lim.split(":")[-1])/60 if int(up_lim.split(":")[-1][0]) != 0 else \
            int(up_lim.split(":")[-1][-1])/60
        start_num = start_hour_num + start_min_num
        final_num = final_hour_num + final_min_num
        first_condition = np.logical_and(time_num >= start_num, time_num <= final_num)
        second_condition = np.round((time_num - start_num)*60)%delta_time == 0.
        file_masked = file_daily[np.logical_and(first_condition, second_condition)]
        return file_masked

    dates_log = []
    start_log = []
    end_log = []
    alpha_log = []
    delta_log = []
    alpha_rate_log = []
    delta_rate_log = []
    for init, final, day_i in zip(init_arr, final_arr, day_arr):
        # Masking day
        day_str_i = str(day_i) if len(str(day_i)) > 1 else "0" + str(day_i)
        file_daily_i = file[np.where(np.char.find(np.char.lower(np.core.defchararray.replace(file.T[0], '2024', '')),
                                                  day_str_i) > -1)[0]]
        final_file = search_condition(file_daily_i, init, final, time_span)
        init_utc_format = [t + ":00UT" for t in final_file.T[1][:-1]]
        final_utc_format = [t + ":00UT" for t in final_file.T[1][1:]]
        date_format = np.repeat("July " + day_str_i, len(final_utc_format)).tolist()
        coord_masked = [apc.SkyCoord(ra, dec, frame='icrs') for ra, dec in zip(final_file.T[2][:-1],
                                                                               final_file.T[3][:-1])]
        alpha_format = [float("{:.4f}".format(coo.ra.value)) for coo in coord_masked]
        delta_format = [float("{:.4f}".format(coo.dec.value)) for coo in coord_masked]
        alpha_rate_format = [float(alpha_rate) for alpha_rate in final_file.T[4][:-1]]
        delta_rate_format = [float(delta_rate) for delta_rate in final_file.T[5][:-1]]
        dates_log.append(date_format)
        start_log.append(init_utc_format)
        end_log.append(final_utc_format)
        alpha_log.append(alpha_format)
        delta_log.append(delta_format)
        alpha_rate_log.append(alpha_rate_format)
        delta_rate_log.append(delta_rate_format)
    df = pd.DataFrame({"Date": list(chain.from_iterable(dates_log)),
                       "Initial UT": list(chain.from_iterable(start_log)),
                       "Final UT": list(chain.from_iterable(end_log)),
                       "alpha": list(chain.from_iterable(alpha_log)),
                       "delta": list(chain.from_iterable(delta_log)),
                       "dRA*cos(D)/dt": list(chain.from_iterable(alpha_rate_log)),
                       "dDEC/dt": list(chain.from_iterable(delta_rate_log))})
    print(df)
    if name_to_save is not None:
        df.to_csv("/home/yiyo/Downloads/" + name_to_save + "_finallog_July17_31.csv", index=False)


def plot_moon(ax_moon, fig_zoom, moon_img, t_start, deltat, data_set=None, site="Paranal Observatory (ESO)",
              telescope="xshoo", mode="offset", drift_zones=None, params=None):
    ax_moon.imshow(moon_img, extent=[-900, 900, -900, 900], alpha=0.72)
    t_end = t_start + deltat
    site = apc.EarthLocation.of_site(site)
    moon1 = apc.get_moon(t_start, location=site)
    moon2 = apc.get_moon(t_end, location=site)
    sun = apc.get_sun(t_start)
    moon_radius_physical = 3478.8 * u.km / 2
    moon_distance = moon1.distance
    moon_radius = ((moon_radius_physical / moon_distance) * 180 * u.deg / np.pi).to(u.arcsec)
    ang = np.linspace(0, 360, 50) * u.deg
    ax_moon.annotate("N", (0, 0.95 * moon_radius.value), ha="center", va="center")
    ax_moon.annotate("E", (0.95 * moon_radius.value, 0), ha="center", va="center")
    ax_moon.annotate("S", (0, -0.95 * moon_radius.value), ha="center", va="center")
    ax_moon.annotate("W", (-0.95 * moon_radius.value, 0), ha="center", va="center")
    ax_moon.plot(moon_radius * np.sin(ang), moon_radius * np.cos(ang))
    ax_moon.arrow(0, 0, ((moon2.ra - moon1.ra) * np.cos(moon1.dec)).to(u.arcsec).value,
                  (moon2.dec - moon1.dec).to(u.arcsec).value, length_includes_head=True, width=10, color='k')
    ax_moon.annotate(f"{int(deltat.value)} min displacement", (0, -100))
    to_sub_earth_angle, position_angle = sub_earth_n_position_angle(sun, moon1)
    post = ')' if to_sub_earth_angle > 90 * u.deg else ''
    pre = '(' if to_sub_earth_angle > 90 * u.deg else ''
    add_label(ax_moon, moon_radius, f"{pre}Sub-solar{post}", to_sub_earth_angle, position_angle, -moon_radius.value/9,
              color="r", ha='center')
    shadow(ax_moon, moon_radius, to_sub_earth_angle, position_angle, color="r")

    ax_moon.annotate(f"{100 - to_sub_earth_angle.to(u.deg).value * 100 / 180:.1f}% illum",
                     (-0.98 * moon_radius.value, 0.9 * moon_radius.value), )
    ax_moon.annotate(f"$\\theta=${-to_sub_earth_angle.to(u.deg).value:.1f} deg",
                     (0.55 * moon_radius.value, 0.9 * moon_radius.value), color='k', )
    t_start_str = t_start.value.split('T')[-1].split(':')[0] + ':' + t_start.value.split('T')[-1].split(':')[1]
    t_end_str = t_end.value.split('T')[-1].split(':')[0] + ':' + t_end.value.split('T')[-1].split(':')[1]
    if data_set is not None:
        path_data, rot_moon = data_set
        pos_zone_moon, pos_center_moon, date_obs_arr = pos_ra_dec(path_data)
        lat, long, coord, x_oblate, y_oblate, z_oblate = from_skyPosition_2_planetPosition(pos_zone_moon,
                                                                                           pos_center_moon, rot_moon,
                                                                                           moon_radius.value,
                                                                                           oblatitude)
        view_moon = point_of_view(coord, rot_moon, moon_radius.value, oblatitude, naxis=3)
        for pos_x, pos_y, t in zip(view_moon[:, 1] * 3600, view_moon[:, 2] * 3600, date_obs_arr):
            if pos_x > 0 and pos_y > 0:
                c = 'aqua'
                idx_ax = 1
            if pos_x > 0 > pos_y:
                c = 'm'
                idx_ax = 0
            if pos_x < 0:
                c = 'k'
                idx_ax = 2
            if t.partition('T')[-1] == '08:00:23.243':
                rect_highlands = Rectangle((pos_x, pos_y - 20), width=15*17, height=40, facecolor=c, alpha=0.4,
                                           label='Highlands (Pilot #1)')
                ax_moon.add_patch(rect_highlands)
            if t.partition('T')[-1] == '08:29:35.498':
                rect_mare = Rectangle((pos_x, pos_y - 20), width=15*15, height=40, facecolor=c, alpha=0.4,
                                      label='Mare Serenitatis (Pilot #2)')
                ax_moon.add_patch(rect_mare)
            if t.partition('T')[-1] == '09:11:16.900':
                rect_dark = Rectangle((pos_x, pos_y - 20), width=40, height=40, facecolor=c, alpha=0.4,
                                      label='Aristarchus (Pilot #3)')
                ax_moon.add_patch(rect_dark)
    if mode == "drift":
        if drift_zones is None:
            raise ValueError("You should provide offsets of the drift zones you want to observe")
        if not np.any(np.array(["extra", "tarot"]) == telescope):
            raise ValueError("Moon drift is only available for ExTrA and Tarot observation strategies")
        m = (moon2.dec - moon1.dec).to(u.arcsec).value/((moon2.ra - moon1.ra)*np.cos(moon1.dec)).to(u.arcsec).value
        tych, apo16, sere, imb, hum, nub, tranqui = [drift_zones[loc] for
                                                     loc in ["tych", "apo16", "sere", "imb", "hum", "nub", "tranqui"]]
        ra_off_tych, dec_off_tych = tych
        ra_off_apo16, dec_off_apo16 = apo16
        ra_off_sere, dec_off_sere = sere
        ra_off_imb, dec_off_imb = imb
        ra_off_hum, dec_off_hum = hum
        ra_off_nub, dec_off_nub = nub
        ra_off_tranqui, dec_off_tranqui = tranqui

        def zoom_plt(axes, offra, offdec, label):
            ap_loc_extra_init = Circle((offra + ((moon2.ra - moon1.ra) * np.cos(moon1.dec)).to(u.arcsec).value,
                                        offdec + (moon2.dec - moon1.dec).to(u.arcsec).value),
                                       radius=10, edgecolor='g', fill=False, alpha=1,
                                       label="Initial fiber position before Moon-drift" if "imb" in label else None)
            ap_loc_extra_end = Circle((offra, offdec), radius=10, edgecolor='r', fill=False, alpha=1,
                                      label="Final fiber position after Moon drift (Pieters et al., 2008)*" if \
                                          "imb" in label else None)
            axes.arrow(offra, offdec, ((moon2.ra - moon1.ra) * np.cos(moon1.dec)).to(u.arcsec).value,
                       (moon2.dec - moon1.dec).to(u.arcsec).value, length_includes_head=True, width=3, color='k')
            axes.imshow(moon_img, extent=[-900, 900, -900, 900], alpha=0.72)
            axes.annotate(label, (offra + 50, offdec + 80), size=12, ha="center",
                          path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
            axes.set_xlim(-900, 900)
            axes.set_ylim(-900, 900)
            axes.add_patch(ap_loc_extra_init)
            axes.add_patch(ap_loc_extra_end)
            shadow(axes, moon_radius, to_sub_earth_angle, position_angle, color="r")
            axes.set_ylabel('Relative arcsec', fontsize=14)
            axes.set_xlabel('Relative arcsec', fontsize=14)
            return [offra + ((moon2.ra - moon1.ra) * np.cos(moon1.dec)).to(u.arcsec).value,
                    offdec + (moon2.dec - moon1.dec).to(u.arcsec).value]

        off_ra_zones = [ra_off_tych, ra_off_apo16, ra_off_sere, ra_off_imb, ra_off_hum, ra_off_nub, ra_off_tranqui]
        off_dec_zones = [dec_off_tych, dec_off_apo16, dec_off_sere, dec_off_imb, dec_off_hum, dec_off_nub,
                         dec_off_tranqui]
        locs = [f"tych: (-43.3°, 348.8°)",
                f"apo16: (-9.0°, 15.5°)",
                f"sere: (25.7°, 18.4°)",
                f"imb: (44.12°, -19.51°)",
                f"hum: (-20.6°, 345.24°)",
                f"nub: (-22.21°, 320.8°)",
                f"tranqui: (8.5°, 31.4°)"]
        extra_init = []
        for ax_zoom, ra_loc, dec_loc, loc in zip(fig_zoom.axes, off_ra_zones, off_dec_zones, locs):
            i_pos = zoom_plt(ax_zoom, ra_loc, dec_loc, loc)
            extra_init.append(i_pos)
        extra_init = np.asarray(extra_init)
        day = int(t_start.value.split("T")[0].split("-")[-1])
        path, r_search, std, guide, Gmag0, Jmag0 = params
        coo_obj = plot_guideStars_and_moon(ax_moon, moon_radius, path, day, t_start, extra_init, r_search, Gmag0=Gmag0,
                                           Jmag0=Jmag0, FOV=29 if telescope == "extra" else 118, std_pos_arr=std,
                                           guide_pos_arr=guide, end=t_end_str)
        pass
    pass


def get_img_from_fig(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format="raw")
    buf.seek(0)
    img_arr = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                         newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1))
    buf.close()
    return img_arr


# rotation values
rotation_values_highlands = np.array([[2.859517, -4.979931, 358.9716],
                                      [2.853309, -4.976692, 358.9663],
                                      [2.850187, -4.975077, 358.9636],
                                      [2.843908, -4.971858, 358.9583],
                                      [2.840750, -4.970253, 358.9557],
                                      [2.834400, -4.967054, 358.9504],
                                      [2.828002, -4.963869, 358.9451],
                                      [2.824786, -4.962281, 358.9425],
                                      [2.821558, -4.960697, 358.9399],
                                      [2.815068, -4.957540, 358.9347],
                                      [2.811805, -4.955967, 358.9321],
                                      [2.808531, -4.954397, 358.9295],
                                      [2.805246, -4.952831, 358.9269],
                                      [2.798642, -4.949710, 358.9217],
                                      [2.791992, -4.946603, 358.9166],
                                      [2.788651, -4.945055, 358.9140],
                                      [2.785298, -4.943511, 358.9115],
                                      [2.775173, -4.938901, 358.9038]])

rotation_values_maria = np.array([[2.764950, -4.934325, 358.8962],
                                  [2.754629, -4.929783, 358.8886],
                                  [2.747695, -4.926775, 358.8836],
                                  [2.740719, -4.923782, 358.8786],
                                  [2.733701, -4.920805, 358.8736],
                                  [2.726641, -4.917844, 358.8687],
                                  [2.723095, -4.916370, 358.8662],
                                  [2.719540, -4.914899, 358.8637],
                                  [2.708812, -4.910512, 358.8563],
                                  [2.701609, -4.907608, 358.8514],
                                  [2.694367, -4.904720, 358.8465],
                                  [2.690730, -4.903282, 358.8441],
                                  [2.683429, -4.900419, 358.8393],
                                  [2.676089, -4.897573, 358.8344],
                                  [2.668710, -4.894744, 358.8296],
                                  [2.661293, -4.891932, 358.8248]])


rotation_values_darkside = np.array([[2.612183, -4.874076, 358.7939],
                                     [2.557650, -4.855693, 358.7614],
                                     [2.533793, -4.848091, 358.7477],
                                     [2.481177, -4.832209, 358.7184]])

####################################################### Data position ################################################
path_uvb = '/home/yiyo/Downloads/ESO_MOON_DATA_XSHOOTER/UVB/SLT_OBJ/'
# obs_pos_moon, planet_pos_moon, date_obs = pos_ra_dec(path_uvb)
rotation_values_moon = np.concatenate((rotation_values_highlands, rotation_values_maria, rotation_values_darkside), 
                                      axis=0)
oblatitude = 1
rotation_values_moon[:, 1] = np.arctan(np.tan(rotation_values_moon[:, 1]*np.pi/180)*(oblatitude**2))*180/np.pi
################################################### Moon image (temporal) ############################################
path_image = '/home/yiyo/Downloads/lroc.png'
img = Image.open(path_image)
img.resize((300, 300))
a = np.asarray(img)
a_corr = ndimage.rotate(a, 180, reshape=False)
#ax.imshow(a_corr, extent=[-900, 900, -900, 900], alpha=0.72)

################################################### ExTrA ############################################
path_extra = '/home/yiyo/Downloads/ephem_moonMay28_June5.csv'
fig = plt.figure(layout='constrained', figsize=(15, 10))
subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[1, 1])

ax = subfigs[0].subplots(1, 1)
axs_loc = subfigs[1].subplots(4, 2)

time1 = apt.Time("2024-12-13T03:35:00", format='isot', scale='utc')
dt_moon_motion = 8*u.min
data = [path_uvb, rotation_values_moon]

zones_to_drift = {"tych": (-309.73, -548.12), "apo16": (211.05, -120.73),
                  "sere": (379.12, 427.43), "imb": (-56.88, 791.04),
                  "hum": (-319.49, -186.35), "nub": (-657.88, -143.31),
                  "tranqui": (528.68, 96.08)}
params_extra = path_extra, 30*u.arcmin, None, None, 9., 6
plot_moon(ax, subfigs[1], a_corr, time1, dt_moon_motion, data_set=data, site="La Silla Observatory (ESO)",
          telescope="extra", mode="drift", drift_zones=zones_to_drift, params=params_extra)

ax.set_xlim(-2800, 2800)
ax.set_ylim(-2800, 2800)
ax.legend(fontsize=7)

subfigs[1].legend(loc="center", fontsize=14)
subfigs[1].subplots_adjust(right=0.85)
plt.show()

########################## Moon eph. for Tarot  #################################
path_moon = '/home/yiyo/Downloads/ephem_TCA_July17_31_1min.csv'
dt_tarot_eph = 5
days_lasilla  =  [17,       18,       18,      19,      19,      20,      20,      21,       22,      23,     24,
                  25,       26,       27,      28,      29,      30]
inits_lasilla = ["23:30", "00:00", "23:30", "00:00", "23:30", "00:00", "23:30", "00:00",  "00:00", "01:30", "02:00",
                 "04:00", "04:00", "06:00", "07:00", "08:00", "08:30"]
ends_lasilla  = ["23:55", "05:00", "23:55", "05:00", "23:55", "08:00", "23:55", "10:00",  "08:00", "10:00", "10:00",
                 "10:00", "10:00", "10:00", "10:00", "10:00", "10:00"]

days_niza  = [17,        18,       19,      19,       20,      20,       21,      21,       22,       22,        23,
              23,        24,       24,      25,       25,      26,       26,      27,       27,       28,        29,
              30,        31]
inits_niza = ["21:30", "21:30",  "21:30", "22:00", "00:00",  "21:30",  "00:00",  "21:30",  "00:00",  "22:00",  "00:00",
              "22:30",  "00:00", "22:30", "00:00", "23:00",  "00:00",  "23:00",  "00:00",  "23:00",  "00:00",  "00:00",
              "00:40",  "01:50"]
ends_niza  = ["22:50", "23:30",  "20:45", "23:55", "00:30",  "23:55",  "01:25",  "23:55",  "01:25",  "23:55",  "01:25",
              "23:55",  "01:25", "23:55", "01:25", "23:55",  "01:25",  "23:55",  "01:25",  "23:55",  "01:25",  "01:25",
              "01:25",  "01:30"]
#tarot_log_to_csv(path_moon, days_niza, inits_niza, ends_niza, dt_tarot_eph, name_to_save="niza")



############################## SNR ##########################################
"""
for H in np.arange(-6, 7)*15:
    angles = np.linspace(0, 360, 100)*u.deg
    yz = []
    for lat in angles:
        coo = np.array([0, 1, 0])
        ncoo = R.from_euler("x", -lat.to(u.deg).value, degrees=True).apply(coo)
        nncoo = R.from_euler("z", -(90 - to_sub_earth_angle.to(u.deg).value + H), degrees=True).apply(ncoo)
        nncoo *= moon_radius.value
        if nncoo[0] > 0:
            yz.append(nncoo[[1, 2]])
            ax.plot(*list(zip(*yz)), 'o', markersize=0.2, c='k')

        coo_label = np.array([0, 1, 0])
        ncoo_label = R.from_euler("x", 10, degrees=True).apply(coo_label)
        nncoo_label = R.from_euler("z", -(90 - to_sub_earth_angle.to(u.deg).value + H), degrees=True).apply(ncoo_label)
        nncoo_label *= moon_radius.value
        if nncoo_label[0] > 0:
            ax.annotate(f"{H/15}", (nncoo_label[1], nncoo_label[2]),)
"""

"""
############################################## MOON + SNR ##########################################################
fig = plt.figure(layout='constrained', figsize=(14, 5))
subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[1, 1])

axMOON = subfigs[0].subplots(1, 1)
pathMOON = '/home/yiyo/Pictures/Moon/XShooter/moon_proposal.png'
imgMOON = Image.open(pathMOON)
#imgMOON.resize((500, 500))
aMOON = np.asarray(imgMOON)
axMOON.imshow(aMOON)
axMOON.set_xticklabels([])
axMOON.set_yticklabels([])

axs_snr = subfigs[1].subplots(3, 1)
for c, region in zip(['cyan', 'r', 'green'], ['highlands', 'maria', 'darkside']):
    for axs, arm, rms_mask, lamba_eff in zip(axs_snr, ['UVB', 'VIS', 'NIR'], lamba_rms_mask, lamba_eff_nd):
        if region == 'highlands' and arm == 'NIR':
            continue
        idx_arr = np.arange(0, 14) if arm == 'NIR' and region == 'maria' else None
        sym = "X" if region == 'highlands' else "d" if region == "maria" else "."
        alpha_zone = 1#.5 if region == "maria" else 1
        s_to_n, wv_eff = rms(region, arm, 4, 1, rms_mask, lamba_eff, mask_all, idx_nd=idx_arr,
                             mode="pre_molecfit" if arm == "UVB" else "post_molecfit")
        if arm == 'UVB':
            if region == 'maria':
                lbl = 'Mare Nubium+Mare Humorum'
            if region == 'highlands':
                lbl = 'Mare Imbrium'
            if region == 'darkside':
                lbl = 'Mare Nectaris'
            axs.plot(lamba_eff, s_to_n, sym, alpha=alpha_zone, label=lbl, markersize=7, c=c)
        else:
            axs.plot(lamba_eff, s_to_n, sym, alpha=alpha_zone, markersize=7, c=c)

        if arm == 'UVB' and region == 'darkside':
            axs.set_yscale('log')
            axs.grid()
        if arm == 'NIR' and region == 'darkside':
            axs.set_xlabel(r'$\lambda$ (nm)', fontsize=13)
            axs.set_yscale('log')
            axs.grid()
        if arm == 'VIS' and region == 'darkside':
            y_l = r'$S/N$'
            axs.set_ylabel(y_l, fontsize=13)
            axs.set_yscale('log')
            axs.grid()
        if arm == 'UVB' and region == 'darkside':
            axs.legend(loc='upper left')

fig.show()
"""