from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np

# set the current time
t = Time('2021-12-07 10:00')

with solar_system_ephemeris.set('jpl'):
	sun = get_body_barycentric_posvel('sun', t)
	mercury = get_body_barycentric_posvel('mercury', t)
	venus = get_body_barycentric_posvel('venus', t)
	earth = get_body_barycentric_posvel('earth', t)
	mars = get_body_barycentric_posvel('mars', t)
	jupiter = get_body_barycentric_posvel('jupiter', t)
	saturn = get_body_barycentric_posvel('saturn', t)
	uranus = get_body_barycentric_posvel('uranus', t)
	neptune = get_body_barycentric_posvel('neptune', t)

sunposition = sun[0]
sunvelocity = sun[1]

mercuryposition = mercury[0]
mercuryvelocity = mercury[1]

venusposition = venus[0]
venusvelocity = venus[1]

earthposition = earth[0]
earthvelocity = earth[1]

marsposition = mars[0]
marsvelocity = mars[1]

jupiterposition = jupiter[0]
jupitervelocity = jupiter[1]

saturnposition = saturn[0]
saturnvelocity = saturn[1]

uranusposition = uranus[0]
uranusvelocity = uranus[1]

neptuneposition = neptune[0]
neptunevelocity = neptune[1]

x_positions = [sunposition.x.to_value(u.AU),mercuryposition.x.to_value(u.AU),venusposition.x.to_value(u.AU)
,earthposition.x.to_value(u.AU),marsposition.x.to_value(u.AU),jupiterposition.x.to_value(u.AU)
,saturnposition.x.to_value(u.AU),uranusposition.x.to_value(u.AU),neptuneposition.x.to_value(u.AU)]

y_positions = [sunposition.y.to_value(u.AU),mercuryposition.y.to_value(u.AU),venusposition.y.to_value(u.AU)
,earthposition.y.to_value(u.AU),marsposition.y.to_value(u.AU),jupiterposition.y.to_value(u.AU)
,saturnposition.y.to_value(u.AU),uranusposition.y.to_value(u.AU),neptuneposition.y.to_value(u.AU)]

z_positions = [sunposition.z.to_value(u.AU),mercuryposition.z.to_value(u.AU),venusposition.z.to_value(u.AU)
,earthposition.z.to_value(u.AU),marsposition.z.to_value(u.AU),jupiterposition.z.to_value(u.AU)
,saturnposition.z.to_value(u.AU),uranusposition.z.to_value(u.AU),neptuneposition.z.to_value(u.AU)]

x_velocities = [sunvelocity.x.to_value(u.AU/u.day),mercuryvelocity.x.to_value(u.AU/u.day),venusvelocity.x.to_value(u.AU/u.day)
,earthvelocity.x.to_value(u.AU/u.day),marsvelocity.x.to_value(u.AU/u.day),jupitervelocity.x.to_value(u.AU/u.day)
,saturnvelocity.x.to_value(u.AU/u.day),uranusvelocity.x.to_value(u.AU/u.day),neptunevelocity.x.to_value(u.AU/u.day)]

y_velocities = [sunvelocity.y.to_value(u.AU/u.day),mercuryvelocity.y.to_value(u.AU/u.day),venusvelocity.y.to_value(u.AU/u.day)
,earthvelocity.y.to_value(u.AU/u.day),marsvelocity.y.to_value(u.AU/u.day),jupitervelocity.y.to_value(u.AU/u.day)
,saturnvelocity.y.to_value(u.AU/u.day),uranusvelocity.y.to_value(u.AU/u.day),neptunevelocity.y.to_value(u.AU/u.day)]

z_velocities = [sunvelocity.z.to_value(u.AU/u.day),mercuryvelocity.z.to_value(u.AU/u.day),venusvelocity.z.to_value(u.AU/u.day)
,earthvelocity.z.to_value(u.AU/u.day),marsvelocity.z.to_value(u.AU/u.day),jupitervelocity.z.to_value(u.AU/u.day)
,saturnvelocity.z.to_value(u.AU/u.day),uranusvelocity.z.to_value(u.AU/u.day),neptunevelocity.z.to_value(u.AU/u.day)]

POSITIONS = np.array((x_positions, y_positions, z_positions)).T 	# AU
POSITIONS -= POSITIONS[0] 		# subtracting the sun initial position
VELOCITIES = np.array((x_velocities, y_velocities, z_velocities)).T	# AU/day
VELOCITIES -= VELOCITIES[0] 	# subtracting the sun initial velocity


if __name__ == '__main__':

	names = ['sun','mercury','venus','earth','mars','jupiter','saturn','uranus','neptune']
	dic = {i: name for i,name in enumerate(names)}
	color_list = sns.color_palette("Paired", n_colors=len(names))
	
	plt.figure()
	scatter = plt.scatter(x_positions,y_positions,c=list(dic.keys()),cmap=ListedColormap(color_list.as_hex()))
	plt.legend(handles=scatter.legend_elements()[0], labels=names)
	plt.xlabel('x [AU]')
	plt.ylabel('y [AU]')
	plt.savefig('plots/x-y.png', dpi=300)
	plt.close()

	plt.scatter(x_positions,z_positions)
	scatter = plt.scatter(x_positions,z_positions,c=list(dic.keys()),cmap=ListedColormap(color_list.as_hex()))
	plt.legend(handles=scatter.legend_elements()[0], labels=names)
	plt.xlabel('x [AU]')
	plt.ylabel('z [AU]')
	plt.savefig('plots/x-z.png', dpi=300)
	plt.close()	