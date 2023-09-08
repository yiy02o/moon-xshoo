import numpy as np
from scipy.spatial.transform import Rotation


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


def point_of_view(coord_obs, rotation_values, equatorial_radius, oblatitude, n):
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
