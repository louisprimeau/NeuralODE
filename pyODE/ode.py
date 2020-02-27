def RKIntegrator(f, t, x0, theta, *args):
    shape = (t.shape[0], x0.shape[0], x0.shape[1])
    images = np.zeros(shape)
    images[0,:,:] = x0
    for i in range(len(t)):
        h = t[i] - t[i - 1];
        k1 = f(images[i - 1,:,:], args, t[i]) * h;
        k2 = f(images[i - 1,:,:] + k1 / 2, args, t[i-1]) * h;
        k3 = f(images[i - 1,:,:] + k2 / 2, args, t[i-1]) * h;
        k4 = f(images[i - 1,:,:] + k3, args, t[i-1] + h) * h;
        images[i,:,:] = images[i-1,:,:] + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    return images
