from mult import mult
from ode import RKIntegrator
import numpy as np

# Adjoint Sensitivity Function
def adjsensitivity(f, theta, t, z_t1, dLdz1):
    dLdt1 = mult(dLdz1, f(z_t1, t, theta))
    s0 = np.concatenate((z_t1.flatten(), dLdz1.flatten(), np.zeros(theta.shape).flatten(), -dLdt1.flatten()))[:,np.newaxis]
    def augdynamics(s, t, theta):
        z = f(s[0:z_t1.size,:].reshape((28,28)), t, theta)
        print(s.shape)
        a = z[0:z_t1.size,:]
        b = -mult(s[1],z[z_t1.size:dLdz1.size + z_t1.size,:])
        c = -mult(s[1],z[dLdz1.size + z_t1.size:dLdz1.size + z_t1.size + theta.size,:])
        d = -mult(s[1], z[dLdz1.size + z_t1.size:dLdz1.size + z_t1.size + theta.size:len(z),:])
        return a, b, c, d
    
    (z_to, dLdz0, dLdtheta, dLdt0) = RKIntegrator(augdynamics, t, s0, theta)
    return dLdtheta


