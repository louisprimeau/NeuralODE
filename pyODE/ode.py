5import numpy as np


# My simple Runge-Kutta integrator.
def RKIntegrator(f, t, x0, theta):
    shp = (t.shape[0], x0.shape[0], x0.shape[1])
    images = np.zeros(shp)
    images[0,:,:] = x0
    for i in range(len(t)):
        h = t[i] - t[i - 1];
        k1 = f(images[i - 1,:,:], t[i], theta) * h;
        k2 = f(images[i - 1,:,:] + k1 / 2, t[i-1], theta) * h;
        k3 = f(images[i - 1,:,:] + k2 / 2, t[i-1], theta) * h;
        k4 = f(images[i - 1,:,:] + k3, t[i-1] + h, theta) * h;
        images[i,:,:] = images[i-1,:,:] + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    return images[images.shape[0]-1,:,:]

# TODO: implement adaptive stepsizing
