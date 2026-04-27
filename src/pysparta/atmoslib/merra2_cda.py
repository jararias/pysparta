
import importlib
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from .merra2_lta import var_attrs, MERRA2LTAAtmosphere

logger.disable(__name__)


class __BaseCDAAtmosphere(
    MERRA2LTAAtmosphere,
    database_path=Path(importlib.resources.files('pysparta.atmoslib')) / 'merra2_lta_data'
):
    pass


class MERRA2CDAAtmosphere(
    __BaseCDAAtmosphere,
    database_path=Path(importlib.resources.files('pysparta.atmoslib')) / 'merra2_lta_data'
):

    def get_variable(self, variable, times, sites=None, regular_grid=None,
                     space_interp='bilinear', time_interp='quadratic'):

        if variable in ('beta', 'pwater'):

            # output grid...
            grid_holder = sites or regular_grid
            lat_out = np.array(grid_holder['latitude'], ndmin=1)
            lon_out = np.array(grid_holder['longitude'], ndmin=1)
            time_out = np.array(times, dtype='datetime64[ns]')

            dims = (['time', 'latitude', 'longitude']
                    if sites is None else
                    ['time', 'location'])

            coords = ({'time': time_out, 'latitude': lat_out, 'longitude': lon_out}
                      if sites is None else
                      {'time': time_out,
                       'location': np.arange(len(lat_out)),
                       'latitude': ('location', lat_out),
                       'longitude': ('location', lon_out)})

            return xr.DataArray(
                data=0.01 if variable == 'beta' else 0.1,
                dims=dims, coords=coords, attrs=var_attrs.get(variable))

        return super().get_variable(variable, times, sites, regular_grid,
                                    space_interp, time_interp)

    def get_atmosphere(self, times, sites=None, regular_grid=None, variables=None,
                       space_interp='bilinear', time_interp='quadratic'):
        return super(MERRA2CDAAtmosphere, self).get_atmosphere(
            times, sites, regular_grid, variables, space_interp, time_interp)

    def _load_array(self, variable):
        return super()._load_array(variable)
