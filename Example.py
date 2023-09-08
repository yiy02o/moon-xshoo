import numpy as np
import matplotlib.pyplot as plt
# import Code.Sky2Planet as S2P
from Sky2Planet import *

obs_pos = np.array([[284.843188, -22.5476],
                    [284.843243, -22.54649],
                    [284.843273, -22.54427],
                    [284.843301, -22.54316],
                    [284.840943, -22.54538],
                    [284.84157, -22.54538],
                    [284.842197, -22.54538],
                    [284.84285, -22.54538],
                    [284.843477, -22.54538]])

planet_pos = np.array([[284.84360, -22.54603],
                       [284.84365, -22.54603],
                       [284.84368, -22.54603],
                       [284.84370, -22.54604],
                       [284.84375, -22.54604],
                       [284.84378, -22.54604],
                       [284.84380, -22.54604],
                       [284.84385, -22.54604],
                       [284.84387, -22.54604]])

rotation_values = np.array([[328.369796, 30.067360, 6.3233],
                            [331.654933, 30.067354, 6.3233],
                            [333.251429, 30.067352, 6.3233],
                            [334.847363, 30.067349, 6.3233],
                            [338.127432, 30.067344, 6.3233],
                            [339.722052, 30.067341, 6.3233],
                            [341.321082, 30.067338, 6.3233],
                            [344.577411, 30.067333, 6.3233],
                            [346.159645, 30.067330, 6.3233]])

equatorial_radius = 16.96030 / 2
oblatitude = 1 #(54.364/60.268)  # polar_radius / equatorial_radius
rotation_values[:, 1] = np.arctan(
    np.tan(rotation_values[:, 1]*np.pi/180)*(oblatitude**2))*180/np.pi

lat, long, coord, x_oblate, y_oblate, z_oblate = from_skyPosition_2_planetPosition(
    obs_pos, planet_pos, rotation_values, equatorial_radius, oblatitude)

# Plot
print(lat)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
ax.set_aspect("auto")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-0.003, 0.003)
ax.set_ylim(-0.003, 0.003)
ax.set_zlim(-0.003, 0.003)

ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], s=50,
           c="red", label="Obs. over planet surface")
ax.plot_wireframe(x_oblate, y_oblate, z_oblate,
                  color="darkgoldenrod", rstride=7, cstride=7, label="Oblato")

plt.legend()
plt.show()


view_pos = point_of_view(
    coord, rotation_values, equatorial_radius, oblatitude, n=3)



theta = np.linspace(0, 2 * np.pi, 100)
x1 = equatorial_radius * np.cos(theta) / 60 / 60
x2 = equatorial_radius * oblatitude * np.sin(theta) / 60 / 60

(fig2, axes2) = plt.subplots(1, 1, figsize=(20, 20))
axes2.plot(x1, x2, c='k', label='Saturno')
axes2.plot(view_pos[:, 1], view_pos[:, 2], ".", c="r", label="Observaciones")
axes2.invert_yaxis()
axes2.set_ylim(-0.0045, 0.0045)
axes2.set_xlim(0.0045, -0.0045)
plt.legend()
plt.show()

