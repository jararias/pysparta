
import inspect
import importlib
import warnings
from pathlib import Path
import concurrent.futures as cf

import blosc2
import numpy as np
import xarray as xr
from loguru import logger
from scipy.interpolate import interp1d

from . import filters
from ._base import BaseAtmosphere
from .interpolate import regrid
from .sandbox import assert_that
from .dataio import (
    read_netcdf,
    save_array,
    update_metadata
)

try:
    xesmf = importlib.import_module('xesmf')
except (ImportError, ModuleNotFoundError):
    xesmf = None

logger.disable(__name__)

var_attrs = {
    'albedo': {
        'description': 'ground albedo',
        'standard_name': 'surface_albedo',
        'units': '1'},
    'pressure': {
        'description': 'atmospheric pressure at ground level',
        'standard_name': 'air_pressure',
        'units': 'hPa'},
    'ozone': {
        'description': 'total-column ozone content',
        'standard_name': 'atmosphere_mass_content_of_ozone',
        'units': 'atm-cm'},
    'pwater': {
        'description': 'precipitable water',
        'standard_name': 'atmosphere_mass_content_of_water_vapor',
        'units': 'atm-cm'},
    'beta': {
        'description': 'aerosol angstrom turbidity',
        'standard_name': 'atmosphere_optical_thickness_due_to_ambient_aerosol_particles',
        'units': '1'},
    'alpha': {
        'description': 'aerosol angstrom exponent',
        'standard_name': 'angstrom_exponent_of_ambient_aerosol_in_air',
        'units': 1},
    'ssa': {
        'description': 'shortwave aerosol single scattering albedo',
        'standard_name': 'single_scattering_albedo_in_air_due_to_ambient_aerosol_particles',
        'units': '1'},
    'elevation': {
        'description': 'ground elevation',
        'standard_name': 'surface_altitude',
        'units': 'm'}
}

def create_lta_dataset(source_path, target_path='lta'):
    VARIABLES = [
        ('albedo', 3),
        ('pressure', 0),
        ('ozone', 3),
        ('pwater', 2),
        ('alpha', 2),
        ('beta', 3),
        ('ssa', 3),
        ('elevation', 0)
    ]

    def get_target_path(**kwargs):
        return target_path.format(**kwargs)

    def get_source_file_name(**kwargs):
        # source_path = Path('/home/jararias/.solarpandas-data/merra2_lta/2010-2021')
        if kwargs['variable'] == 'elevation':
            return Path(source_path) / 'merra2_elevation.nc4'
        return Path(source_path) / 'merra2_{variable}_lta_2010-2021.nc4'.format(**kwargs)

    for variable, precision in VARIABLES:
        logger.info(f'Processing variable `{variable}`')

        file_name = get_source_file_name(variable=variable)
        if not file_name.exists():
            logger.warning(f'missing file `{file_name}`. Skipping')
            continue

        values, xlon, xlat, _ = read_netcdf(file_name, variable)
        values = values[0] if variable == 'elevation' else values

        path = get_target_path()
        save_array(path, variable, values=values, precision=precision)
        update_metadata(path, variable, latitude=xlat, longitude=xlon)


class MERRA2LTAAtmosphere(
    BaseAtmosphere,
    database_path=Path(importlib.resources.files('pysparta.atmoslib')) / 'merra2_lta_data'
):

    def get_variable(self, variable, times, sites=None, regular_grid=None,
                     space_interp='bilinear', time_interp='quadratic'):

        assert_that(
            self.has_variable(variable),
            f'missing variable `{variable}`')

        assert_that(
            sites is not None or regular_grid is not None,
            'sites or regular_grid must be provided')

        assert_that(
            sites is None or regular_grid is None,
            'sites or regular_grid must be provided, but not the two of them')

        assert_that(
            space_interp in ('nearest', 'bilinear', 'conservative'),
            f'unknown spatial interpolation method `{space_interp}`')

        if space_interp == 'conservative' and xesmf is None:
            msg = ('conservative interpolation requires xESMF, which '
                   'is not available. Using bilinear instead')
            warnings.warn(msg, RuntimeWarning)
            space_interp = 'bilinear'

        if space_interp == 'conservative' and sites is not None:
            msg = ('conservative interpolation cannot be used with sites. '
                   'Using bilinear instead')
            warnings.warn(msg, RuntimeWarning)
            space_interp = 'bilinear'

        # output grid...
        grid_holder = sites or regular_grid
        lat_out = np.array(grid_holder['latitude'], ndmin=1)
        lon_out = np.array(grid_holder['longitude'], ndmin=1)
        time_out = np.array(times, dtype='datetime64[ns]')

        # input grid...
        lat_in = self.get_latitudes(variable)
        lon_in = self.get_longitudes(variable)
        years = 1970 + np.unique(time_out.astype('datetime64[Y]')).astype(int)
        time_in = np.array(
            [f'{yr}-{mo:02d}-15' for yr in years for mo in range(1, 13)],
            dtype='datetime64[ns]')

        data_in = np.vstack([self._load_array(variable) for _ in range(len(years))])

        if sites is not None:
            if lat_out.shape != lon_out.shape:
                raise AttributeError('shape mismatch in output latitude and longitude')

            data_out = regrid(lon_in, lat_in, data_in, lon_out, lat_out,
                              method=space_interp)

            # perform the temporal interpolation to the target times...
            kwargs = dict(kind=time_interp, fill_value='extrapolate')
            xi = self._get_fractional_year(time_in)
            x = self._get_fractional_year(time_out)
            data_out = interp1d(xi, data_out, axis=0, **kwargs)(x)

            return xr.DataArray(
                data=data_out, dims=['time', 'location'],
                coords={'time': time_out,
                        'location': np.arange(len(lat_out)),
                        'latitude': ('location', lat_out),
                        'longitude': ('location', lon_out)})

        if regular_grid is not None:

            data_out = np.full((12*len(years), len(lat_out), len(lon_out)), np.nan)
 
            if space_interp == 'conservative':

                def center_to_bounds(a):
                    return np.linspace(a[0], a[-1], len(a)+1) - (a[1] - a[0])/2

                grid_in = {
                    'lat': lat_in, 'lon': lon_in,
                    'lat_b': center_to_bounds(lat_in),
                    'lon_b': center_to_bounds(lon_in)}
                grid_out = {
                    'lat': lat_out, 'lon': lon_out,
                    'lat_b': center_to_bounds(lat_out),
                    'lon_b': center_to_bounds(lon_out)}
                regridder = xesmf.Regridder(grid_in, grid_out, space_interp)
                data_out[:] = regridder(data_in)

            else:
                xlon, xlat = np.meshgrid(lon_out, lat_out)
                data_out[:] = regrid(lon_in, lat_in, data_in, xlon, xlat,
                                     method=space_interp)

            # perform the temporal interpolation to the target times...
            kwargs = dict(kind=time_interp, fill_value='extrapolate')
            xi = self._get_fractional_year(time_in)
            x = self._get_fractional_year(time_out)
            data_out = interp1d(xi, data_out, axis=0, **kwargs)(x)

            return xr.DataArray(
                data=data_out,
                dims=['time', 'latitude', 'longitude'],
                coords={'time': time_out, 'latitude': lat_out, 'longitude': lon_out},
                attrs=var_attrs.get(variable))

        return xr.DataArray(
            data=data_out,
            dims=['time', 'latitude', 'longitude'],
            coords={'time': time_out, 'latitude': lat_out, 'longitude': lon_out},
            attrs=var_attrs.get(variable))

    def get_atmosphere(self, times, sites=None, regular_grid=None, variables=None,
                       space_interp='bilinear', time_interp='quadratic'):

        req_variables = self.variables if variables is None else variables
        with cf.ThreadPoolExecutor(max_workers=len(req_variables)) as executor:

            args = (times, sites, regular_grid, space_interp, time_interp)
            futures = {executor.submit(self.get_variable, variable, *args): variable
                       for variable in req_variables}
            logger.debug('futures submitted!!')

            data = {}
            for future in cf.as_completed(futures):
                variable = futures[future]
                logger.debug(f'variable `{variable}` completed')
                data[variable] = future.result()

        return xr.Dataset(data)

    @staticmethod
    def _get_fractional_year(times):
        one_day = np.timedelta64(1, 'D')
        one_year = np.timedelta64(1, 'Y')
        jan_1st = times.astype('datetime64[Y]').astype('datetime64[D]')
        dec_31st = jan_1st.astype('datetime64[Y]') + one_year - one_day
        year_length = dec_31st - jan_1st + one_day
        year_fraction = (times - jan_1st) / year_length
        return times.astype('datetime64[Y]').astype('f4') + 1970 + year_fraction

    def _load_array(self, variable):
        assert_that(
            variable in self._metadata,
            f'missing variable `{variable}`')

        file_name = self.database_path / f'{variable}.bl2'
        data = blosc2.load_array(file_name.as_posix())

        known_filters = dict(inspect.getmembers(filters, inspect.isclass))
        for filter_descr in self._metadata[variable]['filters']:
            filter_cls_name = filter_descr['class']
            if filter_cls_name not in known_filters:
                raise ValueError(f'unknown filter `{filter_cls_name}`')
            filter_cls = known_filters[filter_cls_name]
            filter_cls_kwargs = filter_descr['kwargs']
            data = filter_cls(**filter_cls_kwargs).decode(data)

        return data
