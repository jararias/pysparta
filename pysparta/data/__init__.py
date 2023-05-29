# flake8: noqa

from .io import save, load
from .merra2 import get_clearsky_atmosphere as get_merra2_clearsky_atmosphere

# from pathlib import Path
# import pandas as pd
# import pysparta
# pysparta.data.save('afile.csv', **atmos)
# atmos = pysparta.data.load('afile.csv')

# import sunwhere
# sp = sunwhere.sites(atmos['data'].index, longitude=atmos['gridcell_lon'],
#                     latitude=atmos['gridcell_lat'])
# pd.DataFrame(index=atmos['data'].index,
#              data=pysparta.SPARTA(
#                  cosz=sp.cosz, ecf=sp.ecf, **atmos['data'], as_dict=True)).plot()
