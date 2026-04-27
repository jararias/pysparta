
from pathlib import Path

from loguru import logger

from .dataio import (
    read_netcdf,
    save_array,
    update_metadata
)


def netcdf_to_blosc2(source_path, target_path='lta'):
    """
    Usage example:
    from pysparta.atmoslib.export import netcdf_to_blosc2
    netcdf_to_blosc2(
        '/home/jararias/.solarpandas-data/merra2_lta/2010-2021',
        '/home/jararias/code/devel/merra2_lta/blosc2'
    )
    """

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
