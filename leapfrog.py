import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from initial import POSITIONS, VELOCITIES

G = const.G.to_value((u.AU**3)/(u.kg*u.day**2))

def Acceleration(t: float, pos: np.ndarray) -> np.ndarray:
    '''
    function for the gravitational acceleration 
    from the Sun (assumes Sun location is at [0,0,0])
    input:
    t : time 
    pos : 3D position of the planet
    returns:
    gravitaional acceleration from the Sun
    AU/day^2
    '''
    return -((G * const.M_sun * pos)/(pos[0]**2 + pos[1]**2 + pos[2]**2)**(3/2) ).to_value()

def LeapFrog(f: callable, y0: np.ndarray, v0: np.ndarray, t0: float, tf: float, h: float) -> tuple:
    '''
    leapfrog solver for second order ODE's
    input:
    f : ODE
    y0 : initial y values
    v0 : initial y' values
    t0 : initial t values
    tf : finial t values
    h : step in t 
    returns :
    t: t values
    y: y values
    v: y' values
    '''
    # initialize
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), 3))
    v = np.zeros((len(t), 3))
    y[0,:], v[0,:] = y0, v0
    
    # solve
    for i in range(1, len(t)):
        v[int(i-1/2),:] = v[i-1] + 1/2 * h * f(t[i-1], y[i-1])
        y[i,:] = y[i-1] + h * v[int(i-1/2)]
        v[i,:] = v[int(i-1/2)] + 1/2 * h * f(t[i], y[i])
    
    return t, y, v


if __name__ == '__main__':
    # get the orbits
    t , y_mercury, v_mercury = LeapFrog(Acceleration, POSITIONS[1], VELOCITIES[1], 0, 200*365.25, .5)
    t , y_venus, v_venus = LeapFrog(Acceleration, POSITIONS[2], VELOCITIES[2], 0, 200*365.25, .5)
    t , y_earth, v_earth = LeapFrog(Acceleration, POSITIONS[3], VELOCITIES[3], 0, 200*365.25, .5)
    t , y_mars, v_mars = LeapFrog(Acceleration, POSITIONS[4], VELOCITIES[4], 0, 200*365.25, .5)
    t , y_jupiter, v_jupiter = LeapFrog(Acceleration, POSITIONS[5], VELOCITIES[5], 0, 200*365.25, .5)
    t , y_saturn, v_saturn = LeapFrog(Acceleration, POSITIONS[6], VELOCITIES[6], 0, 200*365.25, .5)
    t , y_uranus, v_uranus = LeapFrog(Acceleration, POSITIONS[7], VELOCITIES[7], 0, 200*365.25, .5)
    t , y_neptune, v_neptune = LeapFrog(Acceleration, POSITIONS[8], VELOCITIES[8], 0, 200*365.25, .5)

    # save them for next script
    np.save('output/leapfrog.npy',np.array([y_mercury,y_venus,y_earth,y_mars,y_jupiter,y_saturn,y_uranus,y_neptune]))

    # create plots
    fig = plt.figure(figsize=[6.4, 6.4])
    plt.plot(y_mercury[:,0],y_mercury[:,1],label='mercury')
    plt.plot(y_venus[:,0],y_venus[:,1],label='venus')
    plt.plot(y_earth[:,0],y_earth[:,1],label='earth')
    plt.plot(y_mars[:,0],y_mars[:,1],label='mars')
    plt.plot(y_jupiter[:,0],y_jupiter[:,1],label='jupiter')
    plt.plot(y_saturn[:,0],y_saturn[:,1],label='saturn')
    plt.plot(y_uranus[:,0],y_uranus[:,1],label='uranus')
    plt.plot(y_neptune[:,0],y_neptune[:,1],label='neptune')
    plt.legend()
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.savefig('plots/xy-leapfrog.png', dpi=300)
    plt.close()

    fig = plt.figure(figsize=[6.4, 6.4])
    plt.plot(y_mercury[:,0],y_mercury[:,1],label='mercury')
    plt.plot(y_venus[:,0],y_venus[:,1],label='venus')
    plt.plot(y_earth[:,0],y_earth[:,1],label='earth')
    plt.plot(y_mars[:,0],y_mars[:,1],label='mars')
    plt.legend()
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.savefig('plots/xy-leapfrog-zoom.png', dpi=300)
    plt.close()

    fig = plt.figure()
    plt.plot(t,y_neptune[:,2],label='neptune')
    plt.plot(t,y_uranus[:,2],label='uranus')
    plt.plot(t,y_saturn[:,2],label='saturn')
    plt.plot(t,y_jupiter[:,2],label='jupiter')
    plt.plot(t,y_mars[:,2],label='mars')
    plt.plot(t,y_earth[:,2],label='earth')
    plt.plot(t,y_venus[:,2],label='venus')
    plt.plot(t,y_mercury[:,2],label='mercury')
    plt.legend()
    plt.ylabel('z [AU]')
    plt.xlabel('t [days]')
    plt.savefig('plots/z-leapfrog.png', dpi=300)
    plt.close()

    fig = plt.figure()
    plt.plot(t,y_mars[:,2],label='mars')
    plt.plot(t,y_earth[:,2],label='earth')
    plt.plot(t,y_venus[:,2],label='venus')
    plt.plot(t,y_mercury[:,2],label='mercury')
    plt.legend()
    plt.ylim(-1,1)
    plt.ylabel('z [AU]')
    plt.xlabel('t [days]')
    plt.savefig('plots/z-leapfrog-zoom.png', dpi=300)
    plt.close()