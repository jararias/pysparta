# flake8: noqa
# pylint: disable=no-name-in-module

import os
import importlib

from .io import save, load
from .merra2_local import LTADataset as _LTADataset
from .merra2_local import DailyDatasets as _DailyDatasets
from .merra2_remote import get_clearsky_atmosphere as get_merra2_clearsky_atmosphere

merra2_lta = _LTADataset(importlib.resources.files('pysparta.data') / 'merra2_lta')

def merra2_daily(path=None):
    if not (data_path := path or os.environ.get('MERRA2_DAILY_PATH', None)):
        raise ValueError('missing path to MERRA-2 daily data')
    return _DailyDatasets(data_path)
