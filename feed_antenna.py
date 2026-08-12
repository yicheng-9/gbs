import numpy as np
from scipy.constants import c, pi

class FeedAntenna:
    def __init__(self, freq, edge_angle_deg=45.0, edge_taper_db=-12.0):
        self.freq = freq
        self.lam = c / freq
        self.k = 2 * pi / self.lam
        theta = np.deg2rad(edge_angle_deg)
        cos_t = np.cos(theta)
        numerator = 20 * np.log10((1 + cos_t) / 2) - edge_taper_db
        denominator = 20 * self.k * (1 - cos_t) * np.log10(np.e)
        b = numerator / denominator
        self.b = abs(b)
        self.w0 = np.sqrt(2 * self.b / self.k)

    def far_field_amplitude(self, theta):
        cos_t = np.cos(theta)
        amp = np.exp(self.k * self.b * cos_t) * (1 + cos_t) / 2.0
        return amp / np.exp(self.k * self.b)