
from pathlib import Path
import pandas as pd

import sunwhere


DB_ROOT = Path('/home/jararias/Dropbox/proyectos/SPARTA/'
               'sparta_validation/validation_data')


files = list(DB_ROOT.glob('atmos_*.parquet'))

columns_to_drop = [('merra2', col) for col in ('alpha', 'asy', 'beta', 'ssa', 'pwater')]

data = []
for k_file_name, file_name in enumerate(files):
    print(f'{k_file_name+1:3d}: {file_name.name}')
    df = (
        pd.read_parquet(file_name)
          .drop(columns=['asy20', 'ssa20'], level='variable')
          .dropna()
          .drop(columns=columns_to_drop)
          .droplevel('source', axis=1)
          .rename(columns={'asy15': 'asy', 'ssa15': 'ssa'})
    )

    site_year = file_name.stem[6:]
    df_obs = pd.read_parquet(file_name.parent / f'obsrad_{site_year}.parquet')
    df[['ecf', 'cosz']] = df_obs.loc[df.index][['ecf', 'cosz']]
    data.append(df)

data = pd.concat(data, axis=0)
data.to_parquet('aenet_merra2_data.parquet')
print(data.shape)
