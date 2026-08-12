"""
Geometry models — parametric surface definitions.
========================================================================
Factory functions that build RationalBezierSurface instances for specific
scattering geometries (cylindrical plates, spherical caps, etc.).
"""
import numpy as np
from bezier_core import RationalBezierSurface


def build_cylindrical_plate(radius: float, height: float, alpha_deg: float):
    """Degree-(1,2) rational Bezier patch for a cylindrical plate.

    Cylinder axis = z.  Arc in the xy-plane, centred on the +x axis,
    spanning from -alpha/2 to +alpha/2 in angle.

    Parameters
    ----------
    radius : float
        Cylinder radius (metres).
    height : float
        Plate height along z (metres).
    alpha_deg : float
        Total arc angle (degrees).

    Returns
    -------
    RationalBezierSurface
        A degree-(1, 2) rational Bezier patch.
    """
    alpha = np.deg2rad(alpha_deg)
    c = np.cos(alpha / 2.0)
    s = np.sin(alpha / 2.0)

    P_arc_xy = np.array([
        [c,  -s],       # start
        [1.0 / c, 0.0], # middle (tangent intersection)
        [c,   s],       # end
    ]) * radius

    W_arc = np.array([1.0, c, 1.0])

    control_points = np.zeros((2, 3, 3))
    weights = np.zeros((2, 3))
    for i, z in enumerate([-height / 2.0, height / 2.0]):
        control_points[i, :, 0] = P_arc_xy[:, 0]
        control_points[i, :, 1] = P_arc_xy[:, 1]
        control_points[i, :, 2] = z
        weights[i, :] = W_arc

    return RationalBezierSurface(control_points, weights)
