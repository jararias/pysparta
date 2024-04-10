
import io
import sys
import json
import time
import inspect
import linecache
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4
import blosc2
from loguru import logger

from .filters import TruncFilter


logger.disable(__name__)


def save_csv(file_name, data, gridcell_lon, gridcell_lat):
    data.index.name = 'times_utc'
    with io.open(file_name, 'w', encoding='utf-8') as fh:
        fh.write(f'# gridcell_lon={gridcell_lon:.05f}, gridcell_lat={gridcell_lat:.05f}\n')
        data.round(5).to_csv(fh)


def load_csv(file_name):
    coords = linecache.getline(file_name, 1).lstrip('#').strip().split(',')
    gridcell_lon, gridcell_lat = list(map(lambda s: float(s.split('=')[-1]), coords))
    data = pd.read_csv(file_name, skiprows=1, index_col=0, parse_dates=True)
    return {'data': data, 'gridcell_lon': gridcell_lon, 'gridcell_lat': gridcell_lat}


def save_array(path, variable_name, values,
               precision, dtype='f4', astype='i4', compressor=None,
               test_accuracy=True):

    if not (p := Path(path)).exists():
        logger.debug(f'Creating data path `{p}`')
        p.mkdir(parents=True, exist_ok=True)

    trunc = TruncFilter(precision, dtype, astype)
    cparams = {'codec': 'ZSTD', 'clevel': 9}
    cparams.update(compressor or {})
    if isinstance(cparams['codec'], str):
        cparams['codec'] = blosc2.Codec[cparams['codec']]

    kwargs = {'cparams': cparams}

    file_name = p / f'{variable_name}.bl2'
    if file_name.exists():
        file_name.unlink()
        time.sleep(1)  # otherwise, the save_array function sometimes fails!
    file_size = blosc2.save_array(trunc.encode(values), file_name.as_posix(), **kwargs)
    logger.debug(f'data array saved in `{file_name}` ({file_size / 1024: .2f} Kb)')

    cparams['codec'] = cparams['codec'].name
    update_metadata(
        path, variable_name,
        compressor=cparams,
        filters=[
            {'class': 'TruncFilter',
             'kwargs': {'precision': precision, 'dtype': dtype, 'astype': astype}}
        ])

    if test_accuracy is True:
        xvalues = load_array(path, variable_name)
        residue = values - xvalues
        mbe, rmse = np.nanmean(residue), np.nanmean(residue**2)**0.5
        if abs(mobs := np.nanmean(values)) > 1e-4:
            mbe = f'{mbe/mobs:.1%}'
            rmse = f'{rmse/mobs:.1%}'
        else:
            mbe = f'{mbe}'
            rmse = f'{rmse}'
        mad = f'{np.nanmax(np.abs(residue)):10.3e}'
        logger.info(f'Compression accuracy of {variable_name}:  {mbe=}  {rmse=}  Max. Abs. Diff.={mad}')

    return file_size


def load_array(path, variable):
    """do not remove. It is used by save_array to make consistency checks"""
    file_name = Path(path) / f'{variable}.bl2'
    metadata = json.load(open(Path(path) / 'metadata.json', mode='r', encoding='utf-8'))
    data = blosc2.load_array(file_name.as_posix())

    visible_modules = sys.modules[__name__]
    visible_class_names, visible_classes = zip(*inspect.getmembers(visible_modules, inspect.isclass))
    for filter_descr in metadata[variable]['filters']:
        filter_cls_name = filter_descr['class']
        filter_cls_kwargs = filter_descr['kwargs']
        if filter_cls_name not in visible_class_names:
            raise ValueError(f'unknown filter of type `{filter_descr["class"]}`')
        filter_cls = visible_classes[visible_class_names.index(filter_cls_name)]
        cfilter = filter_cls(**filter_cls_kwargs)
        data = cfilter.decode(data)

    return data


def update_metadata(path, variable=None, latitude=None, longitude=None, **options):
    file_name = Path(path) / 'metadata.json'

    if not file_name.exists():
        logger.debug(f'Creating metadata file `{file_name}`')
        json.dump({'Created': time.ctime()}, file_name.open(mode='w', encoding='utf-8'))

    metadata = json.load(file_name.open(mode='r', encoding='utf-8'))

    entry = metadata
    if variable is not None:
        if variable not in metadata:
            metadata[variable] = {'Created': time.ctime()}
        entry = metadata[variable]
        entry['Last updated'] = time.ctime()

    if latitude is not None:
        logger.debug(f'adding entry `latitude` to metadata/{variable or "_root_"}')
        entry['latitude'] = {
            'start': latitude[0].item(),
            'end': latitude[-1].item(),
            'step': np.unique(np.diff(latitude)).item()
        }

    if longitude is not None:
        logger.debug(f'adding entry `longitude` to metadata/{variable or "_root_"}')
        entry['longitude'] = {
            'start': longitude[0].item(),
            'end': longitude[-1].item(),
            'step': np.unique(np.diff(longitude)).item()
        }

    # if there are other options provided...
    for key, value in options.items():
        logger.debug(f'adding entry `{key}` to metadata/{variable or "_root_"}')
        entry[key] = value

    json.dump(metadata, file_name.open(mode='w', encoding='utf-8'))


def read_netcdf(file_name, variable):
    with netCDF4.Dataset(file_name, 'r') as cdf:
        values = np.array(cdf.variables[variable][:], dtype=np.float32)
        lon = np.array(cdf.variables['lon'][:], dtype=np.float32)
        lat = np.array(cdf.variables['lat'][:], dtype=np.float32)
        try:
            time = cdf.variables['time']
            times = netCDF4.num2date(time[:], units=time.units)
        except KeyError:
            times = None
    return values, lon, lat, times
