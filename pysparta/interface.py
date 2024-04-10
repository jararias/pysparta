
import inspect

import sunwhere
import numpy as np
import xarray as xr

from . import modlib, atmoslib
from .sandbox import assert_that


def run_model(model, times=None, sites=None, regular_grid=None, atmos=None,
              sunwhere_kwargs=None, atmos_kwargs=None, **kwargs):

    assert_that(
        isinstance(model, str),
        f'`model`: expected str, got {model.__class__.__name__}',
        TypeError)

    assert_that(
        hasattr(modlib, model),
        f'unknown `model` {model}')

    model_func = getattr(modlib, model)
    model_vars = inspect.getfullargspec(model_func).args

    if 'cosz' in kwargs:
        kwargs['cosz'] = np.array(kwargs['cosz'])

    if 'ecf' in kwargs:
        kwargs['ecf'] = np.array(kwargs['ecf'])

    if not {'cosz', 'ecf'}.issubset(kwargs):
        # If cosz and ecf not in kwargs, run sunwhere
        # To run sunwhere, I need times+sites or times+regular_grid
        assert_that(
            times is not None,
            '`cosz` and/or `ecf` are not provided. `times` is required')

        assert_that(
            sites is not None or regular_grid is not None,
            '`cosz` and/or `ecf` are not provided. `sites` or '
            '`regular_grid` are required')

        assert_that(
            sites is None or regular_grid is None,
            'only one of `cosz` and `ecf` must be provided')

        if sites is None:
            args = (times, regular_grid['latitude'], regular_grid['longitude'])
            sw = sunwhere.regular_grid(*args, **(sunwhere_kwargs or {}))
            _, n_lats, n_lons = sw.cosz.shape
            kwargs.setdefault('cosz', sw.cosz)
            kwargs.setdefault('ecf', sw.ecf.expand_dims(
                dim={'latitude': n_lats, 'longitude': n_lons}, axis=(1, 2)))
        else:
            args = (times, sites['latitude'], sites['longitude'])
            sw = sunwhere.sites(*args, **(sunwhere_kwargs or {}))
            _, n_locs = sw.cosz.shape
            kwargs.setdefault('cosz', sw.cosz)
            kwargs.setdefault('ecf', sw.ecf.expand_dims(
                dim={'location': n_locs}, axis=1))

    if atmos is not None:
        assert_that(
            isinstance(atmos, str),
            f'`atmos`: expected str, got {atmos.__class__.__name__}',
            TypeError)

        assert_that(
            atmos in atmoslib.databases,
            f'unknown `atmos` {atmos}')

        assert_that(
            times is not None,
            '`cosz` and/or `ecf` are not provided. Then, `times` is required')

        assert_that(
            sites is not None or regular_grid is not None,
            '`cosz` and/or `ecf` are not provided. Then, `sites` or '
            '`regular_grid` are required')

        assert_that(
            sites is None or regular_grid is None,
            'only one of `cosz` and `ecf` must be provided')

        atmos_obj = atmoslib.databases[atmos]
        # must retrieve the atmosphere variables valid for this model...
        variables = set(model_vars).intersection(atmos_obj.variables)
        # ...and that are not provided as kwargs...
        variables = variables.difference(kwargs)
        these_kwargs = atmos_kwargs or {}
        if sites is None:
            these_kwargs['regular_grid'] = regular_grid
        else:
            these_kwargs['sites'] = sites
        kwargs.update(
            atmos_obj.get_atmosphere(times, variables=variables, **these_kwargs)
        )

    result = model_func(**kwargs)

    if isinstance(kwargs['cosz'], xr.DataArray):
        dims = kwargs['cosz'].dims
        coords = kwargs['cosz'].coords
        var_attrs = {
            'ghi': {
                'standard_name': 'global horizontal irradiance',
                'units': 'W m-2'},
            'dni': {
                'standard_name': 'direct normal irradiance',
                'units': 'W m-2'},
            'dhi': {
                'standard_name': 'direct horizontal irradiance',
                'units': 'W m-2'},
            'dif': {
                'standard_name': 'diffuse horizontal irradiance',
                'units': 'W m-2'},
            'csi': {
                'standard_name': 'circumsolar irradiance',
                'units': 'W m-2'},
        }
        return xr.Dataset(
            {key: (dims, values, var_attrs.get(key))
             for key, values in result.items()},
            coords=coords)

    return result
