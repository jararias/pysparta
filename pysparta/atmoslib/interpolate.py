
import numpy as np


def regrid(grid_x, grid_y, grid_z, x, y, method='bilinear'):
    """Interpolation along axes (-2,-1) in rank-n grid_z.

    Parameters
    ----------
    grid_x: array-like, rank 1 of shape (N,)
        coordinate values along dimension x of grid_z (axis=-1)
    grid_y: array-like, rank 1 of shape (M,)
        coordinate values along dimension y of grid_z (axis=-2)
    grid_z: array-like, rank n of shape (..., M, N)
    x: array-like, arbitrary shape, typically rank-1 or rank-2 array
        target coordinate values for dimension x. Must have same shape as y
    y: array-like, arbitrary shape, typically rank-1 or rank-2 array
        target coordinate values for dimension y. Must have same shape as x
    method: str
        interpolation method: nearest or bilinear

    Return
    ------
    Interpolated values in an array with shape (..., shape of x and y). For
    instance, if grid_z has dimensions (dfb, slot, latitude, longitude) and x
    and y are rank-2 longitude and latitude arrays, respectively, with shape
    (P, Q), the output array would have shape (dfb, slot, P, Q). In contrast,
    if the new locations were rank-1 arrays with shape (R,), the shape
    of the output array would be (dfb, slot, R). Same comments apply to input
    arrays with shapes (time, latitude, longitude) or (issue_day, cycle,
    lead_hour, latitude, longitude), for instance.
    """
    # transformation to the segment (0,1)x(0,1)
    def normalize(v, grid):
        return (v - grid[0]) / (grid[-1] - grid[0])
    ycoords = normalize(grid_y, grid_y)
    xcoords = normalize(grid_x, grid_x)
    yinterp = normalize(y, grid_y)
    xinterp = normalize(x, grid_x)

    zvalues = grid_z
    if np.ma.is_masked(zvalues):
        zvalues = np.where(zvalues.mask, np.nan, zvalues.data)
    assert zvalues.ndim >= 2, \
        'grid_val must have at least ndim=2. Got {}'.format(zvalues.ndim)

    def clip(k, kmax):
        return np.clip(k, 0, kmax)

    if method == 'nearest':
        jx = np.rint((grid_y.size - 1) * yinterp).astype('int')
        ix = np.rint((grid_x.size - 1) * xinterp).astype('int')
        jx = clip(jx, grid_y.size - 1)
        ix = clip(ix, grid_x.size - 1)
        return zvalues[..., jx, ix]

    elif method == 'bilinear':
        j1 = ((grid_y.size - 1) * yinterp).astype('int')
        i1 = ((grid_x.size - 1) * xinterp).astype('int')
        jmax, imax = grid_y.size - 1, grid_x.size - 1
        Axy = (ycoords[clip(j1 + 1, jmax)] - ycoords[clip(j1, jmax)]) * \
            (xcoords[clip(i1 + 1, imax)] - xcoords[clip(i1, imax)])
        A11 = (ycoords[clip(j1 + 1, jmax)] - yinterp) * \
            (xcoords[clip(i1 + 1, imax)] - xinterp) / Axy
        A12 = (ycoords[clip(j1 + 1, jmax)] - yinterp) * \
            (xinterp - xcoords[clip(i1, imax)]) / Axy
        A21 = (yinterp - ycoords[clip(j1, jmax)]) * \
            (xcoords[clip(i1 + 1, imax)] - xinterp) / Axy
        A22 = (yinterp - ycoords[clip(j1, jmax)]) * \
            (xinterp - xcoords[clip(i1, imax)]) / Axy
        return (zvalues[..., clip(j1, jmax), clip(i1, imax)] * A11 +
                zvalues[..., clip(j1, jmax), clip(i1 + 1, imax)] * A12 +
                zvalues[..., clip(j1 + 1, jmax), clip(i1, imax)] * A21 +
                zvalues[..., clip(j1 + 1, jmax), clip(i1 + 1, imax)] * A22)

    else:
        raise ValueError(f'unknown interpolation method {method}')
