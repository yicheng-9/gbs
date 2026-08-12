import numpy as np

class POIntegrator:
    def __init__(self, positions, normals, areas, k):
        self.positions = np.asarray(positions, dtype=float)
        self.normals   = np.asarray(normals, dtype=float)
        self.areas     = np.asarray(areas, dtype=float)
        self.k         = float(k)

    def far_field_pattern(self, H_inc, theta_deg, phi_deg=0.0):
        theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
        if np.isscalar(phi_deg):
            phi = np.full_like(theta, np.deg2rad(float(phi_deg)))
        else:
            phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
            if phi.shape != theta.shape:
                raise ValueError("phi_deg 必须与 theta_deg 同形状")

        N = theta.size
        sin_t = np.sin(theta)
        S = np.empty((N, 3), dtype=float)
        S[:, 0] = sin_t * np.cos(phi)
        S[:, 1] = sin_t * np.sin(phi)
        S[:, 2] = np.cos(theta)

        dot_n_s = self.normals @ S.T
        mask = dot_n_s > 0

        Js = 2.0 * np.cross(self.normals, H_inc)
        s_dot_J = Js @ S.T
        term = S[None, :, :] * s_dot_J[:, :, None] - Js[:, None, :]

        phase = np.exp(1j * self.k * (self.positions @ S.T))
        integrand = term * (phase[:, :, None] * self.areas[:, None, None])
        integrand[~mask] = 0.0

        integral = np.sum(integrand, axis=0)
        Es = -1j * self.k / (4 * np.pi) * integral
        return Es

    def rcs_from_Es(self, Es):
        return np.sum(np.abs(Es)**2, axis=-1)