
from .interface import run_model


def SPARTA(times=None, sites=None, regular_grid=None, atmos=None,
           sunwhere_kwargs=None, atmos_kwargs=None, **kwargs):

    return run_model('SPARTA', times, sites, regular_grid, atmos,
                     sunwhere_kwargs, atmos_kwargs, **kwargs)


# Hence, to use the models:
# from pysparta import SPARTA
# from pysparta.benchmark import BIRD

# SPARTA(times, sites={'latitude':..., 'longitude':...}, atmos='merra2_lta')
