
import sys
import importlib

from ._base import BaseAtmosphere
from .merra2_cda import MERRA2CDAAtmosphere
from .merra2_lta import MERRA2LTAAtmosphere

databases = {}


def register(name, base_atmosphere):
    global databases
    if issubclass(base_atmosphere, BaseAtmosphere):
        databases.update({name: base_atmosphere})


register('merra2_cda', MERRA2CDAAtmosphere)
register('merra2_lta', MERRA2LTAAtmosphere)

if (sys.version_info.major, sys.version_info.minor) < (3, 10):
    _entry_points = importlib.metadata.entry_points().get('pysparta.atmos', [])
else:
    _entry_points = importlib.metadata.entry_points(group='pysparta.atmos')

for _atmos_db in _entry_points:
    register(_atmos_db.name, _atmos_db.load())
