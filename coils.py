# import PyQt4
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0

pi = np.pi

import matplotlib.pyplot as plt
# from mayavi.mlab import mesh, show


def get_factors(n):
    """return all the factors of n"""
    factors = set()
    for i in range(1, int(n ** (0.5)) + 1):
        if not n % i:
            factors.update((i, n // i))
    return factors


def segments(x_min, x_max, y_min, y_max, N_segments):
    """Find the optimal cartesian grid for splitting up a rectangle of spanning x_min to
    x_max and y_min to y_max into N_segments equal sized segments such that each segment
    is as close to square as possible. This is the same as minimising the surface area
    between segments. Return a list of the midpoints of each segment"""
    size_x = x_max - x_min
    size_y = y_max - y_min
    lowest_surface_area = None
    for n_x in get_factors(N_segments):
        n_y = N_segments // n_x
        surface_area = n_x * size_y + n_y * size_x
        if lowest_surface_area is None or surface_area < lowest_surface_area:
            lowest_surface_area = surface_area
            best_n_x, best_n_y = n_x, n_y
    dx = size_x / best_n_x
    dy = size_y / best_n_y

    midpoints = []
    for x in np.linspace(x_min + dx / 2, x_max - dx / 2, best_n_x):
        for y in np.linspace(y_min + dy / 2, y_max - dy / 2, best_n_y):
            midpoints.append((x, y))
    return midpoints


def _broadcast(r):
    """If r=(x, y, z) is a tuple or list of arrays or scalars, broadcast it to be a
    single array with the list/tuple index corresponding to the first dimension."""
    if not isinstance(r, np.ndarray):
        return np.array(np.broadcast_arrays(*r))
    return r


def field_of_current_loop(r, z, R, I):
    """Compute, in cylindrical coordinates, Br(r, z), Bz(r, z) of a current loop with
    current I and radius R, centred at the origin with normal vector pointing in the z
    direction"""
    k2 = 4 * r * R / (z ** 2 + (R + r) ** 2)
    E_k2 = ellipe(k2)
    K_k2 = ellipk(k2)
    rprime2 = z ** 2 + (r - R) ** 2

    B_r_num = mu_0 * z * I * ((R ** 2 + z ** 2 + r ** 2) / rprime2 * E_k2 - K_k2)
    B_r_denom = 2 * pi * r * np.sqrt(z ** 2 + (R + r ** 2))

    # Some hoop jumping to set B_r = 0 when r = 0 despite the expression having a
    # division by zero in it in when r = 0:
    if isinstance(r, np.ndarray):
        B_r = np.zeros(B_r_denom.shape)
        B_r[r != 0] = B_r_num[r != 0] / B_r_denom[r != 0]
    elif r == 0:
        B_r = 0.0
    else:
        B_r = B_r_num / B_r_denom

    B_z_num = mu_0 * I * ((R ** 2 - z ** 2 - r ** 2) / rprime2 * E_k2 + K_k2)
    B_z_denom = 2 * pi * np.sqrt(z ** 2 + (R + r ** 2))

    B_z = B_z_num / B_z_denom

    return B_r, B_z


def field_of_current_line(r, z, L, I):
    """compute, in cylindrical coordinates, B_phi(r, z) of a current-carrying straight
    wire of length L running from the origin to z = L with current flowing in the +z
    direction."""
    prefactor = mu_0 * I / (4 * pi * r)
    term1 = z / np.sqrt(r ** 2 + z ** 2)
    term2 = (L - z) / np.sqrt(r ** 2 + (L - z) ** 2)
    return prefactor * (term1 + term2)


class CurrentObject(object):
    def __init__(self, r0, zprime, yprime=None, n_turns=1):
        """A current-carrying object with a coordinate system centred at position r0 =
        (x0, y0, z0), with primary axis pointing along zprime = (zprime_x, zprime_y,
        zprime_z) and secondary axis pointing along yprime = (yprime_x, yprime_y,
        yprime_z). These two axes define the orientation of a right handed coordinate
        system (xprime, yprime, zprime) for the object with respect to the lab
        coordinate directions (x, y, z). The two axes do not need to be normalised (they
        will be normalised automatically), but must be orthogonal. if yprime is None
        (perhaps if the object has rotational symmetry such that it doesn't matter), it
        will be chosen randomly. n_turns is an overall multiplier for the current"""
        self.r0 = np.array(r0)
        self.zprime = np.array(zprime) / np.sqrt(np.dot(zprime, zprime))
        if yprime is None:
            # A random vector that is orthogonal to zprime:
            yprime = np.cross(np.random.randn(3), zprime)
        self.yprime = np.array(yprime) / np.sqrt(np.dot(yprime, yprime))

        if not abs(np.dot(self.yprime, self.zprime)) < 1e-10:
            raise ValueError("Primary and secondary axes of object not orthogonal")

        self.xprime = np.cross(self.yprime, self.zprime)

        # Rotation matrix from local frame to lab frame, and its inverse
        self.Q_rot = np.stack([self.xprime, self.yprime, self.zprime], axis=1)
        self.Q_rot_inv = np.linalg.inv(self.Q_rot)
        self.n_turns = n_turns

    def pos_to_local(self, r):
        """Take a point r = (x, y, z) in the lab frame and return rprime = (xprime,
        yprime, zprime) in the local frame of reference of the object."""
        r = _broadcast(r)
        return np.einsum('ij,j...->i...', self.Q_rot_inv, (r.T - self.r0).T)

    def pos_to_lab(self, rprime):
        """Take a point rprime = (xprime, yprime, zprime) in the local frame of the
        object and  return r = (x, y, z) in the lab frame."""
        rprime = _broadcast(rprime)
        return (np.einsum('ij,j...->i...', self.Q_rot, rprime).T + self.r0).T

    def vector_to_local(self, v):
        """Take a vector v = (v_x, v_y, v_z) in the lab frame and return vprime =
        (v_xprime, v_yprime, v_zprime) in the local frame of reference of the object.
        This is different to transforming coordinates as it only rotates the vector, it
        does not translate it."""
        v = _broadcast(v)
        return np.einsum('ij,j...->i...', self.Q_rot_inv, v)

    def vector_to_lab(self, vprime):
        """Take a vector vprime=(v_xprime, v_yprime, v_zprime) in the local frame of the
        object and return v = (v_x, v_y, v_z) in the lab frame. This is different to
        transforming coordinates as it only rotates the vector, it does not translate
        it."""
        vprime = _broadcast(vprime)
        return np.einsum('ij,j...->i...', self.Q_rot, vprime)

    def B(self, r, I):
        """Return the magnetic field at position r=(x, y, z)"""
        # r = _broadcast(r)
        rprime = self.pos_to_local(r)
        return self.vector_to_lab(self.B_local(rprime, I * self.n_turns))

    def local_surfaces(self):
        return []


class Container(CurrentObject):
    def __init__(self, r0=(0, 0, 0), zprime=(0, 0, 1), yprime=None):
        super().__init__(r0=r0, zprime=zprime, yprime=yprime)
        self.children = []

    def add(self, child):
        self.children.append(child)

    def plot(self, **kwargs):
        for child in self.children:
            child.plot(**kwargs)

    def B(self, r, I):
        Bs = []
        for child in self.children:
            Bs.append(child.B(r, I))
        return sum(Bs)

    def show(self, **kwargs):
        surfaces = [self.pos_to_lab(pts) for pts in self.local_surfaces()]
        for child in self.children:
            surfaces.extend(child.pos_to_lab(pts) for pts in child.local_surfaces())
        for x, y, z in surfaces:
            mesh(x, y, z, **kwargs)
        show()


class Loop(CurrentObject):
    def __init__(self, r0, n, R, n_turns=1):
        """Counterclockwise current loop of radius R, centred at r0 = (x0, y0, z0) with
        normal vector n=(nx, ny, nz)"""
        super().__init__(r0=r0, zprime=n, n_turns=n_turns)
        self.R = R

    def B_local(self, rprime, I):
        """Field due to the loop at position rprime=(xprime, yprime, zprime) for current
        I"""
        xprime, yprime, zprime = rprime
        # Expression we need to call is in cylindrical coordinates:
        rho = np.sqrt(xprime ** 2 + yprime ** 2)
        B_rho, B_zprime = field_of_current_loop(rho, zprime, self.R, I)
        phi = np.arctan2(yprime, xprime)
        B_xprime = B_rho * np.cos(phi)
        B_yprime = B_rho * np.sin(phi)
        return np.array([B_xprime, B_yprime, B_zprime])


class Line(CurrentObject):
    def __init__(self, r0, r1, n_turns=1):
        """Current line from r0 = (x0, y0, z0) to r1 = (x1, y1, z1) with current flowing
        from the former to the latter"""
        super().__init__(r0=r0, zprime=np.array(r1) - np.array(r0), n_turns=n_turns)
        self.L = np.sqrt(((np.array(r1) - np.array(r0))**2).sum())

    def B_local(self, rprime, I):
        """Field due to the loop at position rprime=(xprime, yprime, zprime) for current
        I"""
        xprime, yprime, zprime = rprime
        # Expression we need to call is in cylindrical coordinates:
        rho = np.sqrt(xprime ** 2 + yprime ** 2)
        B_phi = field_of_current_line(rho, zprime, self.L, I)
        phi = np.arctan2(yprime, xprime)
        B_xprime = - B_phi * np.sin(phi)
        B_yprime = B_phi * np.cos(phi)
        return np.array([B_xprime, B_yprime, np.zeros_like(B_xprime)])


class RoundCoil(Container, CurrentObject):
    def __init__(
        self, r0, n, R_inner, R_outer, thickness, n_turns=1, cross_sec_segs=12
    ):
        super().__init__(r0=r0, zprime=n)
        self.r0 = np.array(r0, dtype=float)
        self.n = np.array(n, dtype=float)
        self.n /= np.sqrt(np.dot(self.n, self.n))
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.thickness = thickness
        self.n_turns = n_turns
        self.cross_sec_segs = cross_sec_segs

        n_turns_per_seg = self.n_turns / self.cross_sec_segs
        segs = segments(R_inner, R_outer, -thickness / 2, thickness / 2, cross_sec_segs)
        for R, zprime in segs:
            self.add(Loop(self.r0 + zprime * self.n, n, R, n_turns=n_turns_per_seg))

    def local_surfaces(self, **kwargs):
        # Create arrays (in local coordinates) describing surfaces of the coil for
        # plotting:
        theta = np.linspace(-pi, pi, 361)
        zprime = np.array([-self.thickness / 2, self.thickness / 2])
        r = np.array([self.R_inner, self.R_outer])

        surfaces = []

        # Top and bottom caps:
        _theta, _r, = np.meshgrid(theta, r)
        xprime = _r * np.cos(theta)
        yprime = _r * np.sin(theta)

        surfaces.append((xprime, yprime, zprime[0]))
        surfaces.append((xprime, yprime, zprime[1]))

        # Inner and outer edges:
        _theta, _zprime = np.meshgrid(theta, zprime)
        surfaces.append((r[0] * np.cos(theta), r[0] * np.sin(theta), _zprime))
        surfaces.append((r[1] * np.cos(theta), r[1] * np.sin(theta), _zprime))

        return surfaces


# if __name__ == '__main__':
#     coil1 = RoundCoil((0, 0, 0), (0, 0, 1), 0.8, 1.2, 0.4, cross_sec_segs=12)
#     coil2 = RoundCoil(
#         (2, 0, 0), (0, 1, 1), 0.8 / 2, 1.2 / 2, 0.4 / 2, cross_sec_segs=12
#     )
#     container = Container()
#     container.add(coil1)
#     container.add(coil2)
#     container.show()

# x = np.linspace(-10, 10, 100)
# y = 1
# z = 1

# B = field_of_current_line(x, y, z, 0,0,0,1, 1, 1, 1)

# x, y, z  = np.random.randn(3)
# x0, y0, z0  = np.random.randn(3)
# x1, y1, z1  = np.random.randn(3)

# line = Line((x0, y0, z0), (x1, y1, z1))
# Bx, By, Bz = line.B((x, y, z), I=1)
