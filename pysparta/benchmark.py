
from .interface import run_model


def BIRD(times=None, sites=None, regular_grid=None, atmos=None,
         sunwhere_kwargs=None, atmos_kwargs=None, **kwargs):

    return run_model('BIRD', times, sites, regular_grid, atmos,
                     sunwhere_kwargs, atmos_kwargs, **kwargs)

# (and repeat for each benchmark model in modlib)
