from math import sin, floor

import numpy as np
import pytest
from sympy import Float

from devito import (Grid, Operator, Dimension, SparseFunction, SparseTimeFunction,
                    Function, TimeFunction, Eq, Inc,
                    PrecomputedSparseFunction, PrecomputedSparseTimeFunction,
                    MatrixSparseTimeFunction)
from devito.types import Scalar
from examples.seismic import (demo_model, TimeAxis, RickerSource, Receiver,
                              AcquisitionGeometry, Model)
from examples.seismic.acoustic import AcousticWaveSolver
import scipy.sparse


def unit_box(name='a', shape=(11, 11), grid=None):
    """Create a field with value 0. to 1. in each dimension"""
    grid = grid or Grid(shape=shape)
    a = Function(name=name, grid=grid)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[:] = np.meshgrid(*dims)[1]
    return a


def unit_box_time(name='a', shape=(11, 11)):
    """Create a field with value 0. to 1. in each dimension"""
    grid = Grid(shape=shape)
    a = TimeFunction(name=name, grid=grid, time_order=1)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[0, :] = np.meshgrid(*dims)[1]
    a.data[1, :] = np.meshgrid(*dims)[1]
    return a


def points(grid, ranges, npoints, name='points'):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = SparseFunction(name=name, grid=grid, npoint=npoints)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


def time_points(grid, ranges, npoints, name='points', nt=10):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = SparseTimeFunction(name=name, grid=grid, npoint=npoints, nt=nt)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


def a(shape=(11, 11)):
    grid = Grid(shape=shape)
    a = Function(name='a', grid=grid)
    xarr = np.linspace(0., 1., shape[0])
    yarr = np.linspace(0., 1., shape[1])
    a.data[:] = np.meshgrid(xarr, yarr)[1]
    return a


def at(shape=(11, 11)):
    grid = Grid(shape=shape)
    a = TimeFunction(name='a', grid=grid)
    xarr = np.linspace(0., 1., shape[0])
    yarr = np.linspace(0., 1., shape[1])
    a.data[:] = np.meshgrid(xarr, yarr)[1]
    return a


def custom_points(grid, ranges, npoints, name='points'):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    scale = Dimension(name="scale")
    dim = Dimension(name="dim")
    points = SparseFunction(name=name, grid=grid, dimensions=(scale, dim),
                            shape=(3, npoints), npoint=npoints)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


def precompute_linear_interpolation(points, grid, origin):
    """ Sample precompute function that, given point and grid information
        precomputes gridpoints and interpolation coefficients according to a linear
        scheme to be used in PrecomputedSparseFunction.
    """
    gridpoints = [tuple(floor((point[i]-origin[i])/grid.spacing[i])
                        for i in range(len(point))) for point in points]

    interpolation_coeffs = np.zeros((len(points), 2, 2))
    for i, point in enumerate(points):
        for d in range(grid.dim):
            interpolation_coeffs[i, d, 0] = ((gridpoints[i][d] + 1)*grid.spacing[d] -
                                             point[d])/grid.spacing[d]
            interpolation_coeffs[i, d, 1] = (point[d]-gridpoints[i][d]*grid.spacing[d])\
                / grid.spacing[d]
    return gridpoints, interpolation_coeffs


def test_precomputed_interpolation():
    """ Test interpolation with PrecomputedSparseFunction which accepts
        precomputed values for interpolation coefficients
    """
    shape = (101, 101)
    points = [(.05, .9), (.01, .8), (0.07, 0.84)]
    origin = (0, 0)

    grid = Grid(shape=shape, origin=origin)
    r = 2  # Constant for linear interpolation
    #  because we interpolate across 2 neighbouring points in each dimension

    def init(data):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = sin(grid.spacing[0]*i) + sin(grid.spacing[1]*j)
        return data

    m = Function(name='m', grid=grid, initializer=init, space_order=0)

    gridpoints, interpolation_coeffs = precompute_linear_interpolation(points,
                                                                       grid, origin)

    sf = PrecomputedSparseFunction(name='s', grid=grid, r=r, npoint=len(points),
                                   gridpoints=gridpoints,
                                   interpolation_coeffs=interpolation_coeffs)
    eqn = sf.interpolate(m)
    op = Operator(eqn)
    op()
    expected_values = [sin(point[0]) + sin(point[1]) for point in points]
    assert(all(np.isclose(sf.data, expected_values, rtol=1e-6)))


def test_precomputed_interpolation_time():
    """ Test interpolation with PrecomputedSparseFunction which accepts
        precomputed values for interpolation coefficients, but this time
        with a TimeFunction
    """
    shape = (101, 101)
    points = [(.05, .9), (.01, .8), (0.07, 0.84)]
    origin = (0, 0)

    grid = Grid(shape=shape, origin=origin)
    r = 2  # Constant for linear interpolation
    #  because we interpolate across 2 neighbouring points in each dimension

    u = TimeFunction(name='u', grid=grid, space_order=0, save=5)
    for it in range(5):
        u.data[it, :] = it

    gridpoints, interpolation_coeffs = precompute_linear_interpolation(points,
                                                                       grid, origin)

    sf = PrecomputedSparseTimeFunction(name='s', grid=grid, r=r, npoint=len(points),
                                       nt=5, gridpoints=gridpoints,
                                       interpolation_coeffs=interpolation_coeffs)

    assert sf.data.shape == (5, 3)

    eqn = sf.interpolate(u)
    op = Operator(eqn)
    op(time_m=0, time_M=4)

    for it in range(5):
        assert np.allclose(sf.data[it, :], it)


def test_precomputed_injection():
    """Test injection with PrecomputedSparseFunction which accepts
       precomputed values for interpolation coefficients
    """
    shape = (11, 11)
    coords = [(.05, .95), (.45, .45)]
    origin = (0, 0)
    result = 0.25

    # Constant for linear interpolation
    # because we interpolate across 2 neighbouring points in each dimension
    r = 2

    m = unit_box(shape=shape)
    m.data[:] = 0.

    gridpoints, interpolation_coeffs = precompute_linear_interpolation(coords,
                                                                       m.grid, origin)

    sf = PrecomputedSparseFunction(name='s', grid=m.grid, r=r, npoint=len(coords),
                                   gridpoints=gridpoints,
                                   interpolation_coeffs=interpolation_coeffs)

    expr = sf.inject(m, Float(1.))

    Operator(expr)()

    indices = [slice(0, 2, 1), slice(9, 11, 1)]
    assert np.allclose(m.data[indices], result, rtol=1.e-5)

    indices = [slice(4, 6, 1) for _ in coords]
    assert np.allclose(m.data[indices], result, rtol=1.e-5)


@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = unit_box(shape=shape)
    p = points(a.grid, coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    expr = p.interpolate(a)
    Operator(expr)(a=a)

    assert np.allclose(p.data[:], xcoords, rtol=1e-6)


@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate_cumm(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = unit_box(shape=shape)
    p = points(a.grid, coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    p.data[:] = 1.
    expr = p.interpolate(a, increment=True)
    Operator(expr)(a=a)

    assert np.allclose(p.data[:], xcoords + 1., rtol=1e-6)


@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate_time_shift(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    This test verifies the optional time shifting for SparseTimeFunctions
    """
    a = unit_box_time(shape=shape)
    p = time_points(a.grid, coords, npoints=npoints, nt=10)
    xcoords = p.coordinates.data[:, 0]

    p.data[:] = 1.
    expr = p.interpolate(a, u_t=a.indices[0]+1)
    Operator(expr)(a=a)

    assert np.allclose(p.data[0, :], xcoords, rtol=1e-6)

    p.data[:] = 1.
    expr = p.interpolate(a, p_t=p.indices[0]+1)
    Operator(expr)(a=a)

    assert np.allclose(p.data[1, :], xcoords, rtol=1e-6)

    p.data[:] = 1.
    expr = p.interpolate(a, u_t=a.indices[0]+1,
                         p_t=p.indices[0]+1)
    Operator(expr)(a=a)

    assert np.allclose(p.data[1, :], xcoords, rtol=1e-6)


@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate_array(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = unit_box(shape=shape)
    p = points(a.grid, coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    expr = p.interpolate(a)
    Operator(expr)(a=a, points=p.data[:])

    assert np.allclose(p.data[:], xcoords, rtol=1e-6)


@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate_custom(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = unit_box(shape=shape)
    p = custom_points(a.grid, coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    p.data[:] = 1.
    expr = p.interpolate(a * p.indices[0])
    Operator(expr)(a=a)

    assert np.allclose(p.data[0, :], 0.0 * xcoords, rtol=1e-6)
    assert np.allclose(p.data[1, :], 1.0 * xcoords, rtol=1e-6)
    assert np.allclose(p.data[2, :], 2.0 * xcoords, rtol=1e-6)


def test_interpolation_dx():
    """
    Test interpolation of a SparseFunction from a Derivative of
    a Function.
    """
    u = unit_box(shape=(11, 11))
    sf1 = SparseFunction(name='s', grid=u.grid, npoint=1)
    sf1.coordinates.data[0, :] = (0.5, 0.5)

    op = Operator(sf1.interpolate(u.dx))

    assert sf1.data.shape == (1,)
    u.data[:] = 0.0
    u.data[5, 5] = 4.0
    u.data[4, 5] = 2.0
    u.data[6, 5] = 2.0

    op.apply()
    # Exactly in the middle of 4 points, only 1 nonzero is 4
    assert sf1.data[0] == pytest.approx(-20.0)


@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate_indexed(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid. Unlike other tests,
    here we interpolate an expression built using the indexed notation.
    """
    a = unit_box(shape=shape)
    p = custom_points(a.grid, coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    p.data[:] = 1.
    expr = p.interpolate(a[a.grid.dimensions] * p.indices[0])
    Operator(expr)(a=a)

    assert np.allclose(p.data[0, :], 0.0 * xcoords, rtol=1e-6)
    assert np.allclose(p.data[1, :], 1.0 * xcoords, rtol=1e-6)
    assert np.allclose(p.data[2, :], 2.0 * xcoords, rtol=1e-6)


@pytest.mark.parametrize('shape, coords, result', [
    ((11, 11), [(.05, .95), (.45, .45)], 1.),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 0.5)
])
def test_inject(shape, coords, result, npoints=19):
    """Test point injection with a set of points forming a line
    through the middle of the grid.
    """
    a = unit_box(shape=shape)
    a.data[:] = 0.
    p = points(a.grid, ranges=coords, npoints=npoints)

    expr = p.inject(a, Float(1.))

    Operator(expr)(a=a)

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@pytest.mark.parametrize('shape, coords, result', [
    ((11, 11), [(.05, .95), (.45, .45)], 1.),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 0.5)
])
def test_inject_time_shift(shape, coords, result, npoints=19):
    """Test generic point injection testing the x-coordinate of an
    abitrary set of points going across the grid.
    This test verifies the optional time shifting for SparseTimeFunctions
    """
    a = unit_box_time(shape=shape)
    a.data[:] = 0.
    p = time_points(a.grid, ranges=coords, npoints=npoints)

    expr = p.inject(a, Float(1.), u_t=a.indices[0]+1)

    Operator(expr)(a=a, time=1)

    indices = [slice(1, 1, 1)] + [slice(4, 6, 1) for _ in coords]
    indices[1] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)

    a.data[:] = 0.
    expr = p.inject(a, Float(1.), p_t=p.indices[0]+1)

    Operator(expr)(a=a, time=1)

    indices = [slice(0, 0, 1)] + [slice(4, 6, 1) for _ in coords]
    indices[1] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)

    a.data[:] = 0.
    expr = p.inject(a, Float(1.), u_t=a.indices[0]+1, p_t=p.indices[0]+1)

    Operator(expr)(a=a, time=1)

    indices = [slice(1, 1, 1)] + [slice(4, 6, 1) for _ in coords]
    indices[1] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@pytest.mark.parametrize('shape, coords, result', [
    ((11, 11), [(.05, .95), (.45, .45)], 1.),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 0.5)
])
def test_inject_array(shape, coords, result, npoints=19):
    """Test point injection with a set of points forming a line
    through the middle of the grid.
    """
    a = unit_box(shape=shape)
    a.data[:] = 0.
    p = points(a.grid, ranges=coords, npoints=npoints)
    p2 = points(a.grid, ranges=coords, npoints=npoints, name='p2')
    p2.data[:] = 1.
    expr = p.inject(a, p)

    Operator(expr)(a=a, points=p2.data[:])

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@pytest.mark.parametrize('shape, coords, result', [
    ((11, 11), [(.05, .95), (.45, .45)], 1.),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 0.5)
])
def test_inject_from_field(shape, coords, result, npoints=19):
    """Test point injection from a second field along a line
    through the middle of the grid.
    """
    a = unit_box(shape=shape)
    a.data[:] = 0.
    b = Function(name='b', grid=a.grid)
    b.data[:] = 1.
    p = points(a.grid, ranges=coords, npoints=npoints)

    expr = p.inject(field=a, expr=b)
    Operator(expr)(a=a, b=b)

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@pytest.mark.parametrize('shape', [(50, 50, 50)])
def test_position(shape):
    t0 = 0.0  # Start time
    tn = 500.  # Final time
    nrec = 130  # Number of receivers

    # Create model from preset
    model = demo_model('constant-isotropic', spacing=[15. for _ in shape],
                       shape=shape, nbl=10)

    # Derive timestepping from model spacing
    dt = model.critical_dt
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Source and receiver geometries
    src_coordinates = np.empty((1, len(shape)))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = 30.

    rec_coordinates = np.empty((nrec, len(shape)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec_coordinates[:, 1:] = src_coordinates[0, 1:]

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=t0, tn=tn, src_type='Ricker', f0=0.010)
    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, geometry, time_order=2, space_order=4)

    rec, u, _ = solver.forward(save=False)

    # Define source geometry (center of domain, just below surface) with 100. origin
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5 + 100.
    src.coordinates.data[0, -1] = 130.

    # Define receiver geometry (same as source, but spread across x)
    rec2 = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec2.coordinates.data[:, 0] = np.linspace(100., 100. + model.domain_size[0],
                                              num=nrec)
    rec2.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    ox_g, oy_g, oz_g = tuple(o + 100. for o in model.grid.origin)

    rec1, u1, _ = solver.forward(save=False, src=src, rec=rec2,
                                 o_x=ox_g, o_y=oy_g, o_z=oz_g)

    assert(np.allclose(rec.data, rec1.data, atol=1e-5))


def test_edge_sparse():
    """
    Test that interpolation uses the correct point for the edge case
    where the sparse point is at the origin with non rational grid spacing.
    Due to round up error the interpolation would use the halo point instead of
    the point (0, 0) without the factorizaion of the expressions.
    """
    grid = Grid(shape=(16, 16), extent=(225., 225.), origin=(25., 35.))
    u = unit_box(shape=(16, 16), grid=grid)
    u._data_with_outhalo[:u.space_order, :] = -1
    u._data_with_outhalo[:, :u.space_order] = -1
    sf1 = SparseFunction(name='s', grid=u.grid, npoint=1)
    sf1.coordinates.data[0, :] = (25.0, 35.0)

    expr = sf1.interpolate(u)
    subs = {d.spacing: v for d, v in zip(u.grid.dimensions, u.grid.spacing)}
    op = Operator(expr, subs=subs)
    op()
    assert sf1.data[0] == 0


def test_msf_interpolate():
    """ Test interpolation with MatrixSparseTimeFunction which accepts
        precomputed values for interpolation coefficients, but this time
        with a TimeFunction
    """
    shape = (101, 101)
    points = [(.05, .9), (.01, .8), (0.07, 0.84)]
    origin = (0, 0)

    grid = Grid(shape=shape, origin=origin)
    r = 2  # Constant for linear interpolation
    #  because we interpolate across 2 neighbouring points in each dimension

    u = TimeFunction(name='u', grid=grid, space_order=0, save=5)
    for it in range(5):
        u.data[it, :] = it

    gridpoints, interpolation_coeffs = precompute_linear_interpolation(points,
                                                                       grid, origin)

    matrix = scipy.sparse.eye(len(points))

    sf = MatrixSparseTimeFunction(
        name='s', grid=grid, r=r, matrix=matrix, nt=5
    )

    sf.gridpoints.data[:] = gridpoints
    sf.coefficients_x.data[:] = interpolation_coeffs[:, 0, :]
    sf.coefficients_y.data[:] = interpolation_coeffs[:, 0, :]

    assert sf.data.shape == (5, 3)

    eqn = sf.interpolate(u)
    op = Operator(eqn)
    print(op)

    sf.manual_scatter()
    op(time_m=0, time_M=4)
    sf.manual_gather()

    for it in range(5):
        assert np.allclose(sf.data[it, :], it)

    # Now test injection
    u.data[:] = 0

    eqn_inject = sf.inject(field=u, expr=sf)
    op2 = Operator(eqn_inject)
    print(op2)
    op2(time_m=0, time_M=4)

    # There should be 4 points touched for each source point
    # (5, 90), (1, 80), (7, 84) and x+1, y+1 for each
    nzt, nzx, nzy = np.nonzero(u.data)
    assert np.all(np.unique(nzx) == np.array([1, 2, 5, 6, 7, 8]))
    assert np.all(np.unique(nzy) == np.array([80, 81, 84, 85, 90, 91]))
    assert np.all(np.unique(nzt) == np.array([1, 2, 3, 4]))
    # 12 points x 4 timesteps
    assert nzt.size == 48


# @pytest.mark.parametrize('inj', ('s_id', '1 + s_id', 's_id[0, 0, s_id]'))
@pytest.mark.parametrize('inj', ('s_id',))
@pytest.mark.parametrize('shape', [(50, 50, 50)])
@pytest.mark.parametrize('so', (2, 4, 8))
@pytest.mark.parametrize('tn', (20, 40, 100))
def test_decompose_src_to_aligned(shape, so, tn, inj):
    """ Test decomposition of non-aligned source wavelets to equivalent
        aligned to grid points source wavelets
    """

    spacing = (10., 10., 10)
    origin = (0., 0., 0.)

    # Initialize v field
    v = np.empty(shape, dtype=np.float32)
    v[:, :, :int(shape[2]/2)] = 2
    v[:, :, int(shape[2]/2):] = 1

    # Construct model
    model = Model(vp=v, origin=origin, shape=shape, spacing=spacing, space_order=so)

    x, y, z = model.grid.dimensions  # Get dimensions

    t0 = 0  # Simulation starts a t=0
    dt = 1  # model.critical_dt  # Time step from model grid spacing
    tn = tn
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
    src = RickerSource(name='src', grid=model.grid, f0=f0,
                       npoint=9, time_range=time_range)

    # First, position source centrally in all dimensions, then set depth
    stx = 0.125
    ste = 0.9
    stepx = (ste-stx)/int(np.sqrt(src.npoint))

    # Uniform x,y source spread
    src.coordinates.data[:, :2] = \
        np.array(np.meshgrid(np.arange(stx, ste,
                 stepx), np.arange(stx, ste, stepx))).T.reshape(-1, 2) \
        * np.array(model.domain_size[:1])

    src.coordinates.data[:, -1] = 20  # Depth is 20m

    # Get positions affected by sparse operator
    arr = src.gridpoints_all

    # Source ID function to hold unique id for each point affected
    s_id = Function(name='s_id', shape=model.grid.shape, dimensions=model.grid.dimensions,
                    space_order=0, dtype=np.int32)
    s_m = Function(name='s_m', shape=model.grid.shape, dimensions=model.grid.dimensions,
                   space_order=0, dtype=np.int32)

    nzinds = (arr[:, 0], arr[:, 1], arr[:, 2])
    s_id.data[nzinds] = tuple(np.arange(len(nzinds[0])))
    s_m.data[nzinds[0], nzinds[1], nzinds[2]] = 1

    nnz_shape = (model.grid.shape[0], model.grid.shape[1])  # Change only 3rd dim
    nnz = Function(name='nnz', shape=(list(nnz_shape)), dimensions=(x, y),
                   space_order=0, dtype=np.int32)
    nnz.data[:, :] = s_m.data[:, :, :].sum(2)
    inds = np.where(s_m.data == 1.)
    print("Grid - source positions:", inds)
    maxz = len(np.unique(inds[-1]))
    # Change only 3rd dim
    u = TimeFunction(name="u", grid=model.grid, space_order=so, time_order=2)
    sp_zi = Dimension(name='sp_zi')
    sparse_shape = (model.grid.shape[0], model.grid.shape[1], maxz)
    sp_source_mask = Function(name='sp_source_mask', shape=(list(sparse_shape)),
                              dimensions=(x, y, sp_zi), space_order=0, dtype=np.int32)

    # Now holds IDs
    sp_source_mask.data[inds[0], inds[1], :] = tuple(inds[2][:len(np.unique(inds[2]))])
    # seems good

    # Helper dimension to schedule loops of different sizes together
    id_dim = Dimension(name='id_dim')

    time = model.grid.time_dim
    save_src = TimeFunction(name='save_src', shape=(src.shape[0], len(arr)),
                            dimensions=(time, id_dim))

    inj = eval(inj)
    save_src_term = src.inject(field=save_src[time, inj],
                               expr=src * dt**2 / model.m)

    op1 = Operator(save_src_term)
    op1.apply()

    zind = Scalar(name='zind', dtype=np.int32)
    eq0 = Eq(sp_zi.symbolic_max, nnz[x, y] - 1,
             implicit_dims=(time, x, y))
    eq1 = Eq(zind, sp_source_mask[x, y, sp_zi], implicit_dims=(time, x, y, sp_zi))

    # inj_u = source_mask[x, y, zind] * save_src_u[time, source_id[x, y, zind]]
    # Is source_mask needed /
    inj_u = save_src[time, s_id[x, y, zind]]

    t = model.grid.stepping_dim
    eq_u = Inc(u[t, x, y, zind], inj_u, implicit_dims=(time, x, y, sp_zi))

    tteqs = (eq0, eq1, eq_u)
    op = Operator(tteqs)
    op.apply()

    # Assert that first, last as well as other indices are as expected
    assert(s_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]] == 0)
    assert(s_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]] == len(nzinds[0])-1)
    assert(s_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1],
           nzinds[2][len(nzinds[0])-1]] == len(nzinds[0])-1)

    # injection code here

    # Assert that first, last as well as other indices are as expected
    from devito import norm
    assert (src.shape[0] == save_src.shape[0])
    assert (8*src.shape[1] == save_src.shape[1])
    norm1 = norm(u)

    src_term = src.inject(field=u, expr=src * dt**2 / model.m)
    u.data[:] = 0
    op2 = Operator(src_term)
    op2.apply()
    norm2 = norm(u)
    assert np.isclose(norm1, norm2)
    print(norm1)


@pytest.mark.parametrize('inj', ('r_id', '1 + r_id', 'r_id[0, 0, r_id]'))
@pytest.mark.parametrize('shape', [(50, 50, 50)])
@pytest.mark.parametrize('so', (2, 4, 8))
@pytest.mark.parametrize('tn', (20, 40, 80))
def test_decompose_rec_to_aligned(shape, so, tn, inj):
    """ Test decomposition of non-aligned receiver wavelets to equivalent ones
        aligned to grid points receiver wavelets
    """

    spacing = (10., 10., 10)
    origin = (0., 0., 0.)

    # Initialize v field
    v = np.empty(shape, dtype=np.float32)
    v[:, :, :int(shape[2]/2)] = 2
    v[:, :, int(shape[2]/2):] = 1

    # Construct model
    model = Model(vp=v, origin=origin, shape=shape, spacing=spacing, space_order=so)

    x, y, z = model.grid.dimensions  # Get dimensions

    t0 = 0  # Simulation starts a t=0
    dt = 1  # model.critical_dt  # Time step from model grid spacing
    tn = tn
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
    rec = Receiver(name='rec', grid=model.grid, f0=f0,
                   npoint=9, time_range=time_range)

    # First, position source centrally in all dimensions, then set depth
    stx = 0.125
    ste = 0.9
    stepx = (ste-stx)/int(np.sqrt(rec.npoint))

    # Uniform x,y source spread
    rec.coordinates.data[:, :2] = \
        np.array(np.meshgrid(np.arange(stx, ste,
                 stepx), np.arange(stx, ste, stepx))).T.reshape(-1, 2) \
        * np.array(model.domain_size[:1])

    rec.coordinates.data[:, -1] = 20  # Depth is 20m

    # Get positions affected by sparse operator
    arr = rec.gridpoints_all

    # Source ID function to hold unique id for each point affected
    r_id = Function(name='r_id', shape=model.grid.shape, dimensions=model.grid.dimensions,
                    space_order=0, dtype=np.int32)

    nzinds = (arr[:, 0], arr[:, 1], arr[:, 2])
    r_id.data[nzinds] = tuple(np.arange(len(nzinds[0])))

    # Helper dimension to schedule loops of different sizes together
    id_dim = Dimension(name='id_dim')

    time = model.grid.time_dim
    save_rec = TimeFunction(name='save_rec', shape=(rec.shape[0], len(arr)),
                            dimensions=(time, id_dim))

    inj = eval(inj)
    import pdb;pdb.set_trace()
    save_rec_term = rec.interpolate(field=save_rec[time, inj], expr=rec * dt**2 / model.m)

    op1 = Operator(save_rec_term)
    op1.apply()

    # Assert that first, last as well as other indices are as expected
    assert(r_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]] == 0)
    assert(r_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]] == len(nzinds[0])-1)
    assert(r_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1],
           nzinds[2][len(nzinds[0])-1]] == len(nzinds[0])-1)

    # Assert that first, last as well as other indices are as expected
    assert (rec.shape[0] == save_rec.shape[0])
    assert (8*rec.shape[1] == save_rec.shape[1])
