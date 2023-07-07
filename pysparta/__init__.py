# flake8: noqa

import inspect
import importlib
import pandas as pd

from .version import version as __version__  # pylint: disable=import-error

import sunwhere
from .models.sparta import SPARTA
from . import benchmark_models, data


def run(times, lats, lons, model_name='SPARTA', database='merra2_lta', model_kws=None, sunpos_kws=None):

    model_func = SPARTA if model_name == 'SPARTA' else getattr(benchmark_models, model_name)
    model_vars = inspect.getfullargspec(model_func).args

    data_path = importlib.resources.files('pysparta.data')
    lta = data.merra2_local.LTADataset(data_path / 'merra2_lta')
    variables = set(model_vars).intersection(lta.variables)
    atmos = lta.get_atmos(times=times, lats=lats, lons=lons, variables=variables)

    sp = sunwhere.sites(times, latitude=lats, longitude=lons, **(sunpos_kws or {}))
    res = model_func(cosz=sp.cosz, ecf=sp.ecf, as_dict=True, **(atmos | (model_kws or {})))

    # if the input is a time series, the output can be a pandas dataframe
    # if the input is a time series grid, then the output should be a xarray, perhaps
    return pd.DataFrame(index=times, data=res)
