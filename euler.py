import numpy as np
import matplotlib.pyplot as plt
from leapfrog import Acceleration
from initial import POSITIONS, VELOCITIES

def Euler(f: callable, y0: np.ndarray, v0: np.ndarray, t0: float, tf: float, h: float) -> tuple:
    '''
    euler solver for second order ODE's
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
        y[i,:] = y[i-1,:] + h * v[i-1,:]
        v[i,:] = v[i-1,:] + h * f(t[i-1], y[i-1,:])

    return t, y, v

if __name__ == '__main__':
    # get the orbits
    t , y_mercury, v_mercury = Euler(Acceleration, POSITIONS[1], VELOCITIES[1], 0, 200*365.25, .5)
    t , y_venus, v_venus = Euler(Acceleration, POSITIONS[2], VELOCITIES[2], 0, 200*365.25, .5)
    t , y_earth, v_earth = Euler(Acceleration, POSITIONS[3], VELOCITIES[3], 0, 200*365.25, .5)
    t , y_mars, v_mars = Euler(Acceleration, POSITIONS[4], VELOCITIES[4], 0, 200*365.25, .5)
    t , y_jupiter, v_jupiter = Euler(Acceleration, POSITIONS[5], VELOCITIES[5], 0, 200*365.25, .5)
    t , y_saturn, v_saturn = Euler(Acceleration, POSITIONS[6], VELOCITIES[6], 0, 200*365.25, .5)
    t , y_uranus, v_uranus = Euler(Acceleration, POSITIONS[7], VELOCITIES[7], 0, 200*365.25, .5)
    t , y_neptune, v_neptune = Euler(Acceleration, POSITIONS[8], VELOCITIES[8], 0, 200*365.25, .5)

    #load the orbits of the previous script
    leapfrog = np.load('output/leapfrog.npy')

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
    plt.savefig('plots/xy-euler.png', dpi=300)
    plt.close()

    fig = plt.figure(figsize=[6.4, 6.4])
    plt.plot(y_mercury[:,0],y_mercury[:,1],label='mercury')
    plt.plot(y_venus[:,0],y_venus[:,1],label='venus')
    plt.plot(y_earth[:,0],y_earth[:,1],label='earth')
    plt.plot(y_mars[:,0],y_mars[:,1],label='mars')
    plt.legend()
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.savefig('plots/xy-euler-zoom.png', dpi=300)
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
    plt.savefig('plots/z-euler.png', dpi=300)
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
    plt.savefig('plots/z-euler-zoom.png', dpi=300)
    plt.close()

    fig = plt.figure()
    plt.plot(t,leapfrog[0,:,0]-y_mercury[:,0],label='mercury')
    plt.plot(t,leapfrog[1,:,0]-y_venus[:,0],label='venus')
    plt.plot(t,leapfrog[2,:,0]-y_earth[:,0],label='earth')
    plt.plot(t,leapfrog[3,:,0]-y_mars[:,0],label='mars')
    plt.plot(t,leapfrog[4,:,0]-y_jupiter[:,0],label='jupiter')
    plt.plot(t,leapfrog[5,:,0]-y_saturn[:,0],label='saturn')
    plt.plot(t,leapfrog[6,:,0]-y_uranus[:,0],label='uranus')
    plt.plot(t,leapfrog[7,:,0]-y_neptune[:,0],label='neptune')
    plt.legend()
    plt.ylabel('x [AU]')
    plt.xlabel('t [days]')
    plt.savefig('plots/x-leapfrog-euler.png', dpi=300)
    plt.close()
