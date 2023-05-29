# flake8: noqa

import io
import linecache

import pandas as pd


def save(file_name, data, gridcell_lon, gridcell_lat):
    data.index.name = 'times_utc'
    with io.open(file_name, 'w', encoding='utf-8') as fh:
        fh.write(f'# gridcell_lon={gridcell_lon:.05f}, gridcell_lat={gridcell_lat:.05f}\n')
        data.round(5).to_csv(fh)


def load(file_name):
    coords = linecache.getline(file_name, 1).lstrip('#').strip().split(',')
    gridcell_lon, gridcell_lat = list(map(lambda s: float(s.split('=')[-1]), coords))
    data = pd.read_csv(file_name, skiprows=1, index_col=0, parse_dates=True)
    return {'data': data, 'gridcell_lon': gridcell_lon, 'gridcell_lat': gridcell_lat}
