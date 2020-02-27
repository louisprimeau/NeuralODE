from mult import mult

def adjsensitivity(f, theta, t0, t1, z_t1, dLdz1):
    dLdt1 = mult(dLdz1, f(z_t1, t1, theta))
    s0 = (z_t1, dLdz1, numpy.zeros(theta.shape), -dLdt1)
    s0 = augdynamics(f, s0, t, theta)
    return s0

def augdynamics(f, s, t, theta):
    z, dfdz, dfdtheta, dfdt = f(s[0], t, theta)
    return z, -mult(s[1],dfdz), -mult(s[1],dfdtheta), -mult(s[1], dfdt)
