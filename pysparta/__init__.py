# flake8: noqa

from .version import version as __version__  # pylint: disable=import-error

from .models.sparta import SPARTA
from . import benchmark_models, data


# def calculate_clearsky(times, lats, lons, model_name='SPARTA', database='merra2_lta'):
#     import sunwhere

#     sp = sunwhere.sites(times, latitude=lats, longitude=lons)
#     model_ = ... # get model function
#     atmos = ... # search database
#     # perform simulation, and return DataFrame
