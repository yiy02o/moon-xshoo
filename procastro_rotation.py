import os
import procastro as pa
import astropy.coordinates as apc
import astropy.time as apt
import astropy.units as u
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from astropy.io import fits
from PIL import Image
from moon_after_molecfit import *


def from_skyPosition_2_planetPosition(
        obs_pos,
        planet_pos,
        rotation_values,
        equatorial_radius,
        oblatitude,
        n_points=100):
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
    *** The Latitude given by Horizons needs to be transform from 
    Planetodetic to Planetocentric.

    -equatorial_radius: Angular radius of the target given in 
    arcsec. It can be found in Horizons as Ang-diam.
        For example: 16.96030

    -oblatitude: the ration between the planets radius
        For example: Equatorial_radius/North_pole_radius 

    Returns:
    -A plot showing the observations' positions in the sky with 
    respect to the object center and the places in the body of 
    the same observations considering the rotation.
    -Three arrays. The first two contain the Longitude and Latitude
    of the observations over the planet. The last array has the coordinates
    (x, y, z) of the ovservation over the planet
    """
    
    N = len(obs_pos)
    radius = equatorial_radius / (60 * 60)
    radius_a = radius
    radius_b = radius * oblatitude

    yy = (obs_pos[:, 0] - planet_pos[:, 0]) * np.cos(obs_pos[:, 1]*np.pi/180)
    zz = obs_pos[:, 1] - planet_pos[:, 1]
    dist = radius ** 2 - zz ** 2 - yy ** 2

    print("hola no")
    grid_long = np.linspace(-180, 180, n_points)
    grid_lat = np.linspace(-90, 90, n_points)
    grilla = np.meshgrid(grid_long, grid_lat, copy=True,
                        sparse=False, indexing='xy')

    x_coor_rf = radius_a * \
        np.cos(grilla[1] * np.pi/180) * np.cos(grilla[0] * np.pi/180)
    y_coor_rf = radius_a * \
        np.cos(grilla[1] * np.pi/180) * np.sin(grilla[0] * np.pi/180)
    z_coor_rf = radius_b * np.sin(grilla[1] * np.pi/180)

    def find_closest(y_point, z_point, x_coor, y_coor, z_coor):
        closest_dist = radius
        closest_x = 10
        for i in range(len(grid_long)):
            for j in range(len(grid_lat)):
                distance = (y_coor[i][j] - y_point)**2 + \
                    (z_coor[i][j] - z_point)**2
                if distance < closest_dist:
                    if x_coor[i][j] <= 0:
                        closest_x = np.abs(x_coor[i][j])
                        closest_dist = distance
        return closest_x

    rotated_oblato = np.zeros((N, 3, len(x_coor_rf), len(x_coor_rf[0, :])))
    for k in range(N):
        angles = rotation_values[k]
        rotation3 = Rotation.from_euler('zyx', [angles[0],
                                                -angles[1], -angles[2]], degrees=True)
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

        if oblatitude == 1:
            xx = -np.sqrt(np.abs(dist))
            print("hola sis")
        else:
            xx[k] = -find_closest(yy[k], zz[k], x_coor, y_coor, z_coor)

    obs_positions = np.zeros((N, 3))
    origin_frame = np.zeros((N, 3))

    for i in range(N):
        obs_positions[i] = np.array((xx[i], yy[i], zz[i]))
        if dist[i] < 0:
            origin_frame[i] = obs_positions[i]
        else:
            angles = rotation_values[i]

            # rotation of the target in such a way that the point (Lat=0, Lon=0) is pointing to us.
            # as well as the planet rotation axes is align with the earth axes.
            # LONG LAT NP
            rotation1 = Rotation.from_euler('xyz', [angles[2],
                                                    angles[1], -angles[0]], degrees=True)
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

    return (longitude, latitude, origin_frame, x_coor, y_coor, z_coor)


def point_of_view(coord_obs: object, rotation_values: object, equatorial_radius: object, oblatitude: object, n: object) -> object:
    """
    Sky observation ---> Planet observation
    This function transform the (RA, DEC) coordinates of an observation
    to position over a planet surface. The values returned are Longitud and Latitude
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
    *** The Latitude given by Horizons needs to be transform from 
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
    radius_b = radius * oblatitude
    N = len(coord_obs)

    yy = coord_obs[:, 1]
    zz = coord_obs[:, 2]
    dist = radius ** 2 - zz ** 2 - yy ** 2

    returned_positions = np.zeros((N, 3))
    for i in range(N):
        if dist[i] < 0:
            returned_positions[i] = coord_obs[i]
        else:
            angles = rotation_values[n]

            # rotation of the target in such a way that the point (Lat=0, Lon=0) is pointing to us.
            # as well as the planet rotation axes is align with the earth axes.
            # LONG LAT NP
            rotation3 = Rotation.from_euler('zyx', [angles[0],
                                                    -angles[1], -angles[2]], degrees=True)
            returned_positions[i] = rotation3.apply(coord_obs[i])
    return returned_positions


def pos_ra_dec(path):
    pos_obs = []
    moon_center = []
    date_obs = []
    for file in os.listdir(path):
        pathfile = os.path.join(path, file)
        hdul = fits.open(pathfile)
        if hdul[0].header['HIERARCH ESO DPR TYPE'] == 'SKY':  # we will include the sky files soon
            continue
        if hdul[0].header['DATE-OBS'] == '2018-03-10T08:27:22.037':#i == 18:
            #print('aquí esta el malo')
            #i += 1
            continue
        ra_obs, dec_obs = hdul[0].header['RA'], hdul[0].header['DEC']
        ra_off, dec_off = hdul[0].header['HIERARCH ESO SEQ CUMOFF RA'], hdul[0].header['HIERARCH ESO SEQ CUMOFF DEC']
        pos_obs_i = [ra_obs, dec_obs]
        moon_center_i = [ra_obs - ra_off/3600, dec_obs - dec_off/3600]  # offset coordinates are in arcsec
        pos_obs.append(pos_obs_i)
        moon_center.append(moon_center_i)
        date_obs.append(hdul[0].header['DATE-OBS'])
        #hdul.close()
    pos_obs = np.asarray(pos_obs)
    moon_center = np.asarray(moon_center)
    date_obs = np.asarray(date_obs)
    return pos_obs, moon_center, date_obs

'''
# from paranal
obs_pos_highlands = np.array([[269.546018, -19.70124],
                              [269.565750, -19.70485],
                              [269.577711, -19.70663],
                              [269.589437, -19.70836],
                              [269.601164, -19.71009],
                              [269.621604, -19.71386],
                              [269.638041, -19.71670],
                              [269.650947, -19.71870],
                              [269.663146, -19.72054],
                              [269.683589, -19.72431],
                              [269.695553, -19.72610],
                              [269.707517, -19.72788],
                              [269.719011, -19.72955],
                              [269.738514, -19.73310],
                              [269.750715, -19.73494],
                              [269.763152, -19.73683],
                              [269.774883, -19.73856],
                              [269.804048, -19.74439]])

obs_pos_maria= np.array([[269.701760, -19.48580],
                         [269.724762, -19.49018],
                         [269.740708, -19.49291],
                         [269.757830, -19.49591],
                         [269.776365, -19.49924],
                         [269.798900, -19.50352],
                         [269.811084, -19.50535],
                         [269.825621, -19.50775],
                         [269.844864, -19.51125],
                         [269.866932, -19.51541],
                         [269.881941, -19.51791],
                         [269.896951, -19.52041],
                         [269.909609, -19.52236],
                         [269.931680, -19.52653],
                         [269.949515, -19.52970],
                         [269.966410, -19.53264]])

obs_pos_darkside = np.array([[269.745603, -19.59638],
                             [269.840472, -19.61872],
                             [269.885679, -19.62936],
                             [269.978231, -19.65114]])

obs_pos_moon = np.concatenate((obs_pos_highlands, obs_pos_maria, obs_pos_darkside), axis=0)

# center of the moon

planet_pos_highlands = np.array([[269.450000, -19.53310],
                                 [269.462500, -19.53610],
                                 [269.470800, -19.53750],
                                 [269.483300, -19.54060],
                                 [269.487500, -19.54190],
                                 [269.500000, -19.54530],
                                 [269.512500, -19.54810],
                                 [269.520800, -19.54970],
                                 [269.525000, -19.55110],
                                 [269.537500, -19.55390],
                                 [269.545800, -19.55560],
                                 [269.550000, -19.55690],
                                 [269.558300, -19.55830],
                                 [269.570800, -19.56140],
                                 [269.583300, -19.56420],
                                 [269.587500, -19.56580],
                                 [269.595800, -19.56720],
                                 [269.612500, -19.57170]])

'''
'''
planet_pos_maria = np.array([[269.629200, -19.575833],
                             [269.650000, -19.580083],
                             [269.662500, -19.582889],
                             [269.675000, -19.585694],
                             [269.683300, -19.588472],
                             [269.695800, -19.591250],
                             [269.704200, -19.592611],
                             [269.708300, -19.593900],
                             [269.729200, -19.598100],
                             [269.737500, -19.600800],
                             [269.750000, -19.603600],
                             [269.758300, -19.604700],
                             [269.766700, -19.607500],
                             [269.779200, -19.610000],
                             [269.791700, -19.612800],
                             [269.804200, -19.615300]])

planet_pos_darkside = np.array([[269.879200, -19.631900],
                                [269.954200, -19.648900],
                                [269.987500, -19.655800],
                                [270.058300, -19.670300]])

planet_pos_moon = np.concatenate((planet_pos_highlands, planet_pos_maria, planet_pos_darkside), axis=0)

'''
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
                                      #[2.771776, -4.937372, 358.9013]])  # el intruso

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

path_uvb = '/home/yiyo/Downloads/ESO_MOON_DATA_XSHOOTER/UVB/SLT_OBJ/'
obs_pos_moon, planet_pos_moon, date_obs = pos_ra_dec(path_uvb)
rotation_values_moon = np.concatenate((rotation_values_highlands, rotation_values_maria, rotation_values_darkside), 
                                      axis=0)

equatorial_radius_moon = 1795.429 / 2  #16.96030 / 2
oblatitude = 1 #(54.364/60.268)  # polar_radius / equatorial_radius
rotation_values_moon[:, 1] = np.arctan(np.tan(rotation_values_moon[:, 1]*np.pi/180)*(oblatitude**2))*180/np.pi

lat, long, coord, x_oblate, y_oblate, z_oblate = from_skyPosition_2_planetPosition(obs_pos_moon, 
                                                                                   planet_pos_moon, 
                                                                                   rotation_values_moon, 
                                                                                   equatorial_radius_moon,
                                                                                   oblatitude)

# Plot

#fig = plt.figure(figsize=(20, 20))
#axes = fig.add_subplot(projection='3d')
#axes.set_aspect("auto")
#
#axes.set_xlabel('X')
#axes.set_ylabel('Y')
#axes.set_zlabel('Z')
##ax.set_xlim(-0.003, 0.003)
##ax.set_ylim(-0.003, 0.003)
##ax.set_zlim(-0.003, 0.003)
#
#axes.scatter(coord[:, 0], coord[:, 1], coord[:, 2],
#             c="red", label="Obs. over planet surface")
#axes.plot_wireframe(x_oblate, y_oblate, z_oblate,
#                    color="darkgoldenrod", rstride=7, cstride=7, label="Oblato")
#
#plt.legend()
#plt.show()

# %%
fig = plt.figure(layout='constrained', figsize=(15, 5))
subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[1, 1])

ax = subfigs[0].subplots(1, 1)
# subfigs[0].suptitle('Left plots', fontsize='x-large')
#f, ax = pa.figaxes(figsize=(6, 6))
#fig.tight_layout()
ax.set_aspect('equal')

#time = apt.Time("2022-01-02T18:33:00", format='isot', scale='utc') # new Moon
#time = apt.Time("2022-08-09T23:30:00", format='isot', scale='utc')  #N1
#time = apt.Time("2022-08-12T01:00:00", format='isot', scale='utc')  #N2
#time = apt.Time("2022-08-15T06:30:00", format='isot', scale='utc')  #N3
time = apt.Time("2018-03-10T08:00:00", format='isot', scale='utc')
#time = apt.Time("2024-03-04T06:00:00", format='isot', scale='utc')
span_time_hours = 3.6
time2 = time+10*u.min
#site = apc.EarthLocation.of_site("lco")
site = apc.EarthLocation.of_site("Paranal Observatory (ESO)")
moon = apc.get_moon(time, location=site)
moon2 = apc.get_moon(time2, location=site)
sun = apc.get_sun(time)
moon_radius_physical = 3478.8*u.km/2

moon_distance = moon.distance
moon_radius = ((moon_radius_physical / moon_distance) * 180 * u.deg / np.pi).to(u.arcsec)


def add_label(text, to_earth, azimuth, offset_label=-200, ha="left", va="center", **kwargs):
    coordinates = ([(moon_radius * np.sin(to_earth) * np.sin(azimuth)).value,
                    (moon_radius * np.sin(to_earth) * np.cos(azimuth)).value])
    ax.plot(*coordinates, 'o', **kwargs)
    ax.annotate(text, coordinates, (coordinates[0], coordinates[1]+offset_label),
                ha=ha, va=va,
                #arrowprops={'arrowstyle': '->'},
                **kwargs)


def add_label_xy(text, xyz, offset_label, **kwargs):

    ax.annotate(f"{text}{'$_{{bhd}}$' if xyz[2]<0 else ''}", xyz[[0, 1]], (xyz[0]+offset_label[0],
                                                                           xyz[1]+offset_label[1]),
                arrowprops={'arrowstyle': '->'}, **kwargs)


def sub_earth_n_position_angle(body):
    elongation = moon.separation(body)
    print(elongation)
    to_sub_earth_angle = (np.arctan2(body.distance * np.sin(elongation),
                                     moon_distance -
                                     body.distance * np.cos(elongation)))

    print(to_sub_earth_angle.to(u.deg))
    position_angle = moon.position_angle(body)

    return to_sub_earth_angle, position_angle


def shadow(axes, delta, azimuth, **kwargs):

    xy = []
    for angle in np.linspace(0, 360, 50)*u.deg:
        coo = np.array([0, np.cos(angle), np.sin(angle)])*u.deg
        ncoo = R.from_euler("y", -delta.to(u.deg).value, degrees=True).apply(coo)
        nncoo = R.from_euler("x", -azimuth.to(u.deg).value, degrees=True).apply(ncoo)
        nncoo *= moon_radius.value
        if nncoo[0] > 0:
            xy.append(nncoo[[1, 2]])

    axes.plot(*list(zip(*xy)), **kwargs)
    print(*xy)


def xyz_from_sidereal(skycoord):

    coo = skycoord.cartesian.get_xyz()
    ncoo = R.from_euler("z", moon.ra.to(u.deg).value, degrees=True).apply(coo)
    nncoo = R.from_euler("y", -moon.dec.to(u.deg).value, degrees=True).apply(ncoo)

    return nncoo[[1, 2, 0]]*moon_radius.value


def plot_parallactic_angle(time, span, delta, correction_deg=30):
    par = np.array([moon_radius.value*1.02, moon_radius.value*1.08])
    times = time + np.arange(0, span*60, delta.value)*u.min
    idx_color = (1*u.hour/delta).to(u.dimensionless_unscaled).value

    mn = apc.get_moon(times, site)
    altaz = mn.transform_to(apc.AltAz(obstime=times, location=site))
    ha = times.sidereal_time("mean", longitude=site.lon) - mn.ra
    par_ang = np.arctan2(np.sin(ha)*np.cos(site.lat),
                         (np.cos(mn.dec)*np.sin(site.lat) - np.cos(site.lat)*np.sin(mn.dec)*np.cos(ha)))
    # par_ang = np.arcsin(np.sin(ha)*np.cos(site.lon)/np.cos(altaz.alt))
    # par_ang = np.arcsin(np.sin(altaz.az)*np.cos(site.lon)/np.cos(mn.dec))

    print(f"lon{90*u.deg-site.lon} dec{90*u.deg-mn.dec}")
    for t, h, a, z, p in zip(times, ha, altaz.alt, altaz.az, par_ang.to(u.deg)):
        print(t, h, a, z, p)
    print(f"lon{90*u.deg-site.lon} dec{90*u.deg-mn.dec}")

    for ang, t in zip(par_ang+correction_deg*u.deg, times):
        plt.plot(par*np.sin(ang), par*np.cos(ang), color="blue")
        tm = t.ymdhms
        if tm[4] == 0:
            ax.plot(par[1] * np.sin(ang).value, par[1] * np.cos(ang).value, "or")
            ax.annotate(f"{tm[3]}", (1.04 * par[1] * np.sin(ang).value, 1.04 * par[1] * np.cos(ang).value),
                                    size=7, ha='center', va='center', rotation=-ang.to(u.deg).value)

    col = np.arange(len(times)) % idx_color == 0
    print(idx_color)
    plt.plot(par[1]*np.sin(par_ang[0]+correction_deg*u.deg),
             par[1]*np.cos(par_ang[0]+correction_deg*u.deg),
             "o", color="black", markersize=5)


angles = np.linspace(0, 360, 50)*u.deg
ax.plot(moon_radius*np.sin(angles), moon_radius*np.cos(angles))
# ax.plot(1800*np.sin(angles), 1800*np.cos(angles))
ax.arrow(0, 0, ((moon2.ra-moon.ra)*np.cos(moon.dec)).to(u.arcsec).value, (moon2.dec-moon.dec).to(u.arcsec).value,
         length_includes_head=True, width=20)
ax.annotate("10min displacement", (0, -100))

to_sub_earth_angle, position_angle = sub_earth_n_position_angle(sun)
post = ')' if to_sub_earth_angle > 90*u.deg else ''
pre = '(' if to_sub_earth_angle > 90*u.deg else ''
add_label(f"{pre}Sub-solar{post}",
          to_sub_earth_angle, position_angle, -moon_radius.value/9, color="red", ha='center')
shadow(ax, to_sub_earth_angle, position_angle, color="red")

#ax.annotate(f"R$_L$ {moon_radius.value:.1f}''",
#            (0.98*moon_radius.value, 0.9*moon_radius.value),
#            ha="right")
ax.annotate(f"{100-to_sub_earth_angle.to(u.deg).value*100/180:.1f}% illum",
            (-0.98*moon_radius.value, 0.9*moon_radius.value),
            )
ax.annotate(f"$\\theta=${to_sub_earth_angle.to(u.deg).value:.1f} deg",
            (0.55*moon_radius.value, 0.9*moon_radius.value), color='k',
            )

#ax.annotate(f"$\\alpha$ {moon.ra.to(u.hourangle).value:.1f}$^h$",
#            (-0.95*moon_radius.value, -0.95*moon_radius.value),
#            )
#ax.annotate(f"$\\delta$ {'+' if moon.dec.value > 0 else ''}"
#            f"{moon.dec.to(u.deg).value:.1f}$^\circ$",
#            (0.95*moon_radius.value, -0.95*moon_radius.value),
#            ha="right")


#add_label_xy(f"radiant", xyz_from_sidereal(apc.SkyCoord("3h +58d00")), (-moon_radius.value/5,
                                                                        #-moon_radius.value/5))

path_image = '/home/yiyo/Downloads/lroc.png'
img = Image.open(path_image)
img.resize((300, 300))
a = np.asarray(img)
ax.imshow(a, extent=[-900, 900, -900, 900], alpha=0.72)

# ax.set_title(time, fontsize=18)

ax.annotate("N", (0, 0.95*moon_radius.value), ha="center", va="center")
ax.annotate("E", (0.95*moon_radius.value, 0), ha="center", va="center")
ax.annotate("S", (0, -0.95*moon_radius.value), ha="center", va="center")
ax.annotate("W", (-0.95*moon_radius.value, 0), ha="center", va="center")


view_pos_moon = point_of_view(coord, rotation_values_moon, equatorial_radius_moon, oblatitude, n=3)
#view_pos_moon = point_of_view(coord, rotation_values_moon, moon_radius.value, oblatitude, n=3)

theta = np.linspace(0, 2 * np.pi, 100)
x1 = equatorial_radius_moon * np.cos(theta) #/ 60 / 60
x2 = equatorial_radius_moon * oblatitude * np.sin(theta) #/ 60 / 60
#x1 = moon_radius.value * np.cos(theta) #/ 60 / 60
#x2 = moon_radius.value * oblatitude * np.sin(theta) #/ 60 / 60

#(fig2, axes2) = plt.subplots(1, 1, figsize=(15, 15))
#ax.plot(x1, x2, c='k', label='Saturno')
axs_loc = subfigs[1].subplots(3, 1)
dates_posImb_newXshoo = ['08:03:37.261', '08:07:56.363', '08:11:57.865', '08:16:21.868', '08:20:35.121']
dates_posNub_newXshoo = ['08:33:51.071', '08:40:12.425', '08:44:51.959', '08:50:21.659', '08:55:54.888']
dates_posImb_extra = dates_posImb_newXshoo[1:4]
dates_posNub_extra = dates_posNub_newXshoo[1:4]
x_tych_init, y_tych_init = 150, 558
for x, y, date in zip(view_pos_moon[:, 1]*3600, view_pos_moon[:, 2]*3600, date_obs):
    if x > 0 and y > 0:
        c = 'aqua'
        idx_ax = 1
    if x > 0 and y < 0:
        c = 'y'
        idx_ax = 0
    if x < 0:
        c = 'k'
        idx_ax = 2
    if date.partition('T')[-1] == '08:00:23.243':
        rect_imb = Rectangle((x, y - 20), width=15*17, height=40, facecolor=c, alpha=0.7, label='Mare Imbrium')
        ax.add_patch(rect_imb)
    if date.partition('T')[-1] == '08:29:35.498':
        rect_nub = Rectangle((x, y - 20), width=15*15, height=40, facecolor=c, alpha=0.7, label='Mare Nubium')
        ax.add_patch(rect_nub)
    if date.partition('T')[-1] == '09:11:16.900':
        rect_fec = Rectangle((x, y - 20), width=40, height=40, facecolor=c, alpha=0.7, label='Mare Fecundidatis')
        ax.add_patch(rect_fec)
        # tycho crater
        rect_tych = Rectangle((x_tych_init, y_tych_init - 18), width=4*45, height=40, facecolor='m', alpha=0.7,
                              label='Tycho crater')
        ax.add_patch(rect_tych)
    # we are not going to plot the nightime of the Moon (just one single position)
    if idx_ax == 2:
        continue
    rect = Rectangle((x - 0.2, y - 5.5), width=0.4, height=11, facecolor='lime', alpha=1)
    # circle = Circle((x, y), radius=2, edgecolor=c, fill=False, linewidth=1)
    axs_loc[idx_ax].add_patch(rect)
    # axs_loc[idx_ax].add_patch(circle)
    is_in = np.logical_or(date.partition('T')[-1] in dates_posImb_newXshoo,
                          date.partition('T')[-1] in dates_posNub_newXshoo)
    if is_in:
        is_in_in = np.logical_or(date.partition('T')[-1] in dates_posImb_extra,
                                 date.partition('T')[-1] in dates_posNub_extra)
        slit_newXshoo = Rectangle((x - 0.2, y - 5.5), width=0.4, height=11, facecolor='orange', alpha=1)
        axs_loc[idx_ax].add_patch(slit_newXshoo)
        if is_in_in:
            aperture_extra = circle = Circle((x, y), radius=2, edgecolor='blue', fill=False, alpha=1)
            axs_loc[idx_ax].add_patch(aperture_extra)

# ax.set_xlim(-1800, 1800)
# ax.set_ylim(-1800, 1800)
off_tych = [45*x for x in np.arange(0, 5)]
for off in off_tych:
    slit_tychXshoo = Rectangle((x_tych_init + off, y_tych_init), width=0.4, height=11, facecolor='orange', alpha=1)
    axs_loc[2].add_patch(slit_tychXshoo)
    if off in off_tych[::2]:#off_tych[1:4]:
        ap_tych_extra = circle = Circle((x_tych_init + off + 0.2, y_tych_init + 5.5), radius=2, edgecolor='blue',
                                        fill=False, alpha=1)
        axs_loc[2].add_patch(ap_tych_extra)


ax.set_xlabel('Relative arcsec', fontsize=14)
ax.set_ylabel('Relative arcsec', fontsize=14)
ax.legend(fontsize=7, loc='lower left')

axs_loc[0].imshow(a, extent=[-900, 900, -900, 900], alpha=0.72)
axs_loc[1].imshow(a, extent=[-900, 900, -900, 900], alpha=0.72)
axs_loc[2].imshow(a, extent=[-900, 900, -900, 900], alpha=0.72)
shadow(axs_loc[1], to_sub_earth_angle, position_angle, color="red")
shadow(axs_loc[2], to_sub_earth_angle, position_angle, color="red")

axs_loc[0].set_xlim(260, 540)
axs_loc[1].set_xlim(110, 340)
axs_loc[2].set_xlim(130, 340)   # tycho crater
axs_loc[0].set_ylim(-615, -585)
axs_loc[1].set_ylim(335, 365)
axs_loc[2].set_ylim(545, 580)   # tycho crater
axs_loc[1].set_ylabel('Relative arcsec', fontsize=14)
axs_loc[2].set_xlabel('Relative arcsec', fontsize=14)
axs_loc[0].set_title('Mare Imbrium', fontsize=16)
axs_loc[1].set_title('Mare Nubium', fontsize=16)
# axs_loc[2].set_title('Mare Fecundidatis', fontsize=16)
axs_loc[2].set_title('Tycho crater', fontsize=16)

# plot_parallactic_angle(time, span_time_hours, 5*u.min)
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

#axs_snr = subfigs[1].subplots(3, 1)
#for c, region in zip(['y', 'aqua', 'k'], ['highlands', 'maria', 'darkside']):
#    for axs, arm, rms_mask, lamba_eff in zip(axs_snr, ['UVB', 'VIS', 'NIR'], lamba_rms_mask, lamba_eff_nd):
#        if region == 'highlands' and arm == 'NIR':
#            continue
#        idx_arr = np.arange(0, 14) if arm == 'NIR' and region == 'maria' else None
#        s_to_n, wv_eff = rms(region, arm, 4, 1, rms_mask, lamba_eff, mask_all, idx_arr=idx_arr)
#        if arm == 'UVB':
#            if region == 'maria':
#                lbl = 'Mare Nubium'
#            if region == 'highlands':
#                lbl = 'Mare Imbrium'
#            if region == 'darkside':
#                lbl = 'Mare Fecundidatis'
#            axs.plot(lamba_eff, s_to_n, '+', label=lbl, markersize=7, c=c)
#        else:
#            axs.plot(lamba_eff, s_to_n, '+', markersize=7, c=c)
#
#        if arm == 'UVB' and region == 'darkside':
#            axs.set_yscale('log')
#            axs.grid()
#        if arm == 'NIR' and region == 'darkside':
#            axs.set_xlabel(r'$\lambda$ (nm)', fontsize=13)
#            axs.set_yscale('log')
#            axs.grid()
#        if arm == 'VIS' and region == 'darkside':
#            y_l = r'$S/N$'
#            axs.set_ylabel(y_l, fontsize=13)
#            axs.set_yscale('log')
#            axs.grid()
#        if arm == 'UVB' and region == 'darkside':
#            axs.legend(loc='upper left')

plt.show()

