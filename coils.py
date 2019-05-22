import PyQt4
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0

pi = np.pi

import matplotlib.pyplot as plt
from mayavi.mlab import mesh, plot3d, show


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
    def __init__(self, r0, zprime, xprime=None, n_turns=1):
        """A current-carrying object with a coordinate system centred at position r0 =
        (x0, y0, z0), with primary axis pointing along zprime = (zprime_x, zprime_y,
        zprime_z) and secondary axis pointing along xprime = (xprime_x, xprime_y,
        xprime_z). These two axes define the orientation of a right handed coordinate
        system (xprime, yprime, zprime) for the object with respect to the lab
        coordinate directions (x, y, z). The two axes do not need to be normalised (they
        will be normalised automatically), but must be orthogonal. if xprime is None
        (perhaps if the object has rotational symmetry such that it doesn't matter), it
        will be chosen randomly. n_turns is an overall multiplier for the current."""
        self.r0 = np.array(r0)
        self.zprime = np.array(zprime) / np.sqrt(np.dot(zprime, zprime))
        if xprime is None:
            # A random vector that is orthogonal to zprime:
            xprime = np.cross(np.random.randn(3), zprime)
        self.xprime = np.array(xprime) / np.sqrt(np.dot(xprime, xprime))

        if not abs(np.dot(self.xprime, self.zprime)) < 1e-10:
            raise ValueError("Primary and secondary axes of object not orthogonal")

        self.yprime = np.cross(self.zprime, self.xprime)

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

    def surfaces(self):
        return [self.pos_to_lab(pts) for pts in self.local_surfaces()]

    def lines(self):
        return [self.pos_to_lab(pts) for pts in self.local_paths()]

    def local_surfaces(self):
        return []

    def local_paths(self):
        return []

    def show(self, surfaces=False, **kwargs):
        if surfaces:
            surfaces = self.surfaces()
            for x, y, z in surfaces:
                mesh(x, y, z, **kwargs)
        lines = self.lines()
        for x, y, z in lines:
            plot3d(x, y, z)
        
        show()


class Container(CurrentObject):
    def __init__(self, r0=(0, 0, 0), zprime=(0, 0, 1), xprime=None, n_turns=1):
        super().__init__(r0=r0, zprime=zprime, xprime=xprime, n_turns=n_turns)
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

    def surfaces(self):
        surfaces = super().surfaces()
        for child in self.children:
            surfaces.extend(child.surfaces())
        return surfaces

    def lines(self):
        lines = [self.pos_to_lab(pts) for pts in self.local_paths()]
        for child in self.children:
            lines.extend(child.lines())
        return lines


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

    def local_paths(self):
        theta = np.linspace(-pi, pi, 361)
        xprime = self.R * np.cos(theta)
        yprime = self.R * np.sin(theta)
        zprime = 0
        return [(xprime, yprime, zprime)]


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

    def local_paths(self):
        zprime = np.linspace(0, self.L)
        xprime = yprime = 0
        return [(xprime, yprime, zprime)]


class Arc(Container, CurrentObject):
    def __init__(self, r0, n, n_perp, R, phi_0, phi_1, n_turns=1, n_seg=12):
        """Current arc forming part of a loop centred at r0 with normal vector n, from
        angle theta_0 to theta_1 defined with respect to the direction n_perp, which
        should be an axis perpendicular to n. Current is flowing from phi_0 to phi_1,
        which if phi_0 < phi_1, is in the positive sense with respect to the normal
        direction n. This arc will is constructed out of n_seg separate line segments,
        so the accuracy can be increased by increasing n_seg."""
        super().__init__(r0=r0, zprime=n, xprime=n_perp, n_turns=n_turns)

        delta_phi = (phi_1 - phi_0) / n_seg
        for i in range(n_seg):
            phi_seg_start = phi_0 + i * delta_phi
            phi_seg_stop = phi_0 + (i + 1) * delta_phi
            xprime0 = R * np.cos(phi_seg_start)
            yprime0 = R * np.sin(phi_seg_start)
            xprime1 = R * np.cos(phi_seg_stop)
            yprime1 = R * np.sin(phi_seg_stop)

            r0_seg = self.pos_to_lab((xprime0, yprime0, 0))
            r1_seg = self.pos_to_lab((xprime1, yprime1, 0))
            self.add(Line(r0_seg, r1_seg, n_turns=n_turns))


class RoundCoil(Container, CurrentObject):
    def __init__(
        self, r0, n, R_inner, R_outer, height, n_turns=1, cross_sec_segs=12
    ):
        """A round loop of conductor with rectangular cross section, centred at r0 with
        normal vector n, inner radius R_inner, outer radius R_outer, and the given
        height (in the normal direction). The finite cross-section is approximated using
        a number cross_sec_segs of 1D current loops distributed evenly through the cross
        section. n_turns is an overall multiplier for the current used in field
        calculations"""
        super().__init__(r0=r0, zprime=n, n_turns=n_turns)
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.height = height
        self.cross_sec_segs = cross_sec_segs

        n_turns_per_seg = self.n_turns / self.cross_sec_segs
        segs = segments(R_inner, R_outer, -height / 2, height / 2, cross_sec_segs)
        for R, zprime in segs:
            r0_loop = self.pos_to_lab((0, 0, zprime))
            self.add(Loop(r0_loop, n, R, n_turns=n_turns_per_seg))

    def local_surfaces(self):
        # Create arrays (in local coordinates) describing surfaces of the coil for
        # plotting:
        theta = np.linspace(-pi, pi, 361)
        zprime = np.array([-self.height / 2, self.height / 2])
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


class StraightSegment(Container, CurrentObject):
    def __init__(self, r0, r1, n, width, height, n_turns=1, cross_sec_segs=12):
        """A straight segment of conductor, with current flowing in a rectangular cross
        section centred on the line from r0 to r1. A vector n normal to the direction of
        current flow determines which direction the 'height' refers to, the width refers
        to the size of the conductor in the remaining direction. The finite
        cross-section is approximated using a number cross_sec_segs of 1D current lines
        distributed evenly through the cross section. n_turns is an overall multiplier
        for the current used in field calculations"""
        r0 = np.array(r0, dtype=float)
        r1 = np.array(r1, dtype=float)
        super().__init__(r0=r0, zprime=r1 - r0, xprime=n, n_turns=n_turns)
        self.width = width
        self.height = height
        self.cross_sec_segs = cross_sec_segs
        self.L = np.sqrt(((np.array(r1) - np.array(r0))**2).sum())

        n_turns_per_seg = self.n_turns / self.cross_sec_segs
        segs = segments(-width / 2, width / 2, -height / 2, height / 2, cross_sec_segs)
        for xprime, yprime in segs:
            r0_line = self.pos_to_lab((xprime, yprime, 0))
            r1_line = self.pos_to_lab((xprime, yprime, self.L))
            self.add(Line(r0_line, r1_line, n_turns=n_turns_per_seg))

    def local_surfaces(self):
        # Create arrays (in local coordinates) describing surfaces of the segment for
        # plotting:
        xprime = np.array([-self.width / 2, self.width / 2])
        yprime = np.array([-self.height / 2, self.height / 2])
        zprime = np.array([0, self.L])

        surfaces = []
        # Top and bottom surfaces:
        _xprime, _zprime = np.meshgrid(xprime, zprime)
        surfaces.append((_xprime, yprime[0], _zprime))
        surfaces.append((_xprime, yprime[1], _zprime))

        # Left and right surfaces:
        _yprime, _zprime = np.meshgrid(yprime, zprime)
        surfaces.append((xprime[0], _yprime, _zprime))
        surfaces.append((xprime[1], _yprime, _zprime))
        return surfaces

if __name__ == '__main__':
    coil1 = RoundCoil((0, 0, 0), (0, 0, 1), 0.8, 1.2, 0.4, cross_sec_segs=12)
    coil2 = RoundCoil(
        (2, 0, 0), (0, 1, 1), 0.8 / 2, 1.2 / 2, 0.4 / 2, cross_sec_segs=12
    )
    arc = Arc((0,1,0), (0,0,1), (1, 0, 0), 1, 0, pi)
    # arc.show()

    straight_seg = StraightSegment((4, 0, 0), (3, 0, 0), (0,1,0), .4, .4)
    straight_seg.show(surfaces=True)

    # container = Container()
    # container.add(arc)
    # container.add(coil1)
    # container.add(coil2)
    # container.add(straight_seg)
    # container.show(surfaces=True)

# x = np.linspace(-10, 10, 100)
# y = 1
# z = 1

# B = field_of_current_line(x, y, z, 0,0,0,1, 1, 1, 1)

# x, y, z  = np.random.randn(3)
# x0, y0, z0  = np.random.randn(3)
# x1, y1, z1  = np.random.randn(3)

# line = Line((x0, y0, z0), (x1, y1, z1))
# Bx, By, Bz = line.B((x, y, z), I=1)
