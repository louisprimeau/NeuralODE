from mult import mult
from ode import RKIntegrator
import numpy as np
from autograd import grad

# Adjoint Sensitivity Function
def adjsensitivity(f, theta, t, z_t1, dLdz1):
    dLdt1 = mult(dLdz1, f(z_t1, t, theta))
    s0 = np.concatenate((z_t1.flatten(), dLdz1.flatten(), np.zeros(theta.shape).flatten(), -dLdt1.flatten()))[:,np.newaxis]
    def augdynamics(z, a, t, theta):
        z = f(s[0:z_t1.size,:].reshape((28,28)), t, theta)
    (z_to, dLdz0, dLdtheta, dLdt0) = RKIntegrator(augdynamics, t, s0, theta)
    return dLdtheta


