# pylint: disable=consider-using-f-string

import sys
import json
import time
import inspect
from pathlib import Path
from collections import defaultdict
import concurrent.futures as cf

import numpy as np
from loguru import logger
from scipy.interpolate import interp1d
import blosc2


logger.disable(__name__)


class TruncFilter:
    def __init__(self, precision, dtype='f4', astype='i4'):
        self._fill_value = -999
        self._data_type = dtype
        self._store_type = astype
        self._precision = int(precision)
        self._scale_factor = 10**self._precision

    def encode(self, data):
        values = self._scale_factor * np.round(data.astype(self._data_type), self._precision)
        return np.where(np.isnan(values), self._fill_value, values).astype(self._store_type)

    def decode(self, data):
        return (np.where(data == self._fill_value, np.nan, data) / self._scale_factor).astype(self._data_type)


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

    cparams['codec'] = cparams['codec'].name  # pylint: disable=no-member
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
    import netCDF4
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


def regrid(grid_x, grid_y, grid_z, x, y, method='bilinear'):
    """Interpolation along axes (-2,-1) in rank-n grid_z.

    Parameters
    ----------
    grid_x: array-like, rank 1 of shape (N,)
        coordinate values along dimension x of grid_z (axis=-1)
    grid_y: array-like, rank 1 of shape (M,)
        coordinate values along dimension y of grid_z (axis=-2)
    grid_z: array-like, rank n of shape (..., M, N)
    x: array-like, arbitrary shape, typically rank-1 or rank-2 array
        target coordinate values for dimension x. Must have same shape as y
    y: array-like, arbitrary shape, typically rank-1 or rank-2 array
        target coordinate values for dimension y. Must have same shape as x
    method: str
        interpolation method: nearest or bilinear

    Return
    ------
    Interpolated values in an array with shape (..., shape of x and y). For
    instance, if grid_z has dimensions (dfb, slot, latitude, longitude) and x
    and y are rank-2 longitude and latitude arrays, respectively, with shape
    (P, Q), the output array would have shape (dfb, slot, P, Q). In contrast,
    if the new locations were rank-1 arrays with shape (R,), the shape
    of the output array would be (dfb, slot, R). Same comments apply to input
    arrays with shapes (time, latitude, longitude) or (issue_day, cycle,
    lead_hour, latitude, longitude), for instance.
    """
    # transformation to the segment (0,1)x(0,1)
    def normalize(v, grid):
        return (v - grid[0]) / (grid[-1] - grid[0])
    ycoords = normalize(grid_y, grid_y)
    xcoords = normalize(grid_x, grid_x)
    yinterp = normalize(y, grid_y)
    xinterp = normalize(x, grid_x)

    zvalues = grid_z
    if np.ma.is_masked(zvalues):
        zvalues = np.where(zvalues.mask, np.nan, zvalues.data)
    assert zvalues.ndim >= 2, \
        'grid_val must have at least ndim=2. Got {}'.format(zvalues.ndim)

    def clip(k, kmax):
        return np.clip(k, 0, kmax)

    if method == 'nearest':
        jx = np.rint((grid_y.size - 1) * yinterp).astype('int')
        ix = np.rint((grid_x.size - 1) * xinterp).astype('int')
        jx = clip(jx, grid_y.size - 1)
        ix = clip(ix, grid_x.size - 1)
        return zvalues[..., jx, ix]

    elif method == 'bilinear':
        j1 = ((grid_y.size - 1) * yinterp).astype('int')
        i1 = ((grid_x.size - 1) * xinterp).astype('int')
        jmax, imax = grid_y.size - 1, grid_x.size - 1
        Axy = (ycoords[clip(j1 + 1, jmax)] - ycoords[clip(j1, jmax)]) * \
            (xcoords[clip(i1 + 1, imax)] - xcoords[clip(i1, imax)])
        A11 = (ycoords[clip(j1 + 1, jmax)] - yinterp) * \
            (xcoords[clip(i1 + 1, imax)] - xinterp) / Axy
        A12 = (ycoords[clip(j1 + 1, jmax)] - yinterp) * \
            (xinterp - xcoords[clip(i1, imax)]) / Axy
        A21 = (yinterp - ycoords[clip(j1, jmax)]) * \
            (xcoords[clip(i1 + 1, imax)] - xinterp) / Axy
        A22 = (yinterp - ycoords[clip(j1, jmax)]) * \
            (xinterp - xcoords[clip(i1, imax)]) / Axy
        return (zvalues[..., clip(j1, jmax), clip(i1, imax)] * A11 +
                zvalues[..., clip(j1, jmax), clip(i1 + 1, imax)] * A12 +
                zvalues[..., clip(j1 + 1, jmax), clip(i1, imax)] * A21 +
                zvalues[..., clip(j1 + 1, jmax), clip(i1 + 1, imax)] * A22)

    else:
        raise ValueError(f'unknown interpolation method {method}')


def create_lta_dataset(target_path='lta'):
    VARIABLES = [('albedo', 3), ('pressure', 0), ('ozone', 3), ('pwater', 2),
                 ('alpha', 2), ('beta', 3), ('ssa', 3), ('elevation', 0)]

    def get_target_path(**kwargs):
        return target_path.format(**kwargs)

    def get_source_file_name(**kwargs):
        db_root = Path('/home/jararias/.solarpandas-data/merra2_lta/2010-2021')
        if kwargs['variable'] == 'elevation':
            return db_root / 'merra2_elevation.nc4'
        return db_root / 'merra2_{variable}_lta_2010-2021.nc4'.format(**kwargs)

    for variable, precision in VARIABLES:
        logger.info(f'Processing variable `{variable}`')

        file_name = get_source_file_name(variable=variable)
        if not file_name.exists():
            logger.warning(f'missing file `{file_name}`. Skipping')

        path = get_target_path()

        values, lon, lat, _ = read_netcdf(file_name, variable)
        values = values[0] if variable == 'elevation' else values

        save_array(path, variable, values=values, precision=precision)
        update_metadata(path, variable, latitude=lat, longitude=lon)


class LTADataset:
    def __init__(self, path):
        self._path = Path(path)
        metadata_file_name = self._path / 'metadata.json'
        if not metadata_file_name.exists():
            raise ValueError(f'missing required file `{metadata_file_name}`')
        self._metadata = json.load(metadata_file_name.open())

        self._variables = [key for key, value in self._metadata.items()
                           if key != 'elevation' and isinstance(value, dict)]

    @property
    def variables(self):
        return self._variables

    def has_variable(self, variable):
        return variable in self.variables

    def get_latitude(self, variable):
        if (not self.has_variable(variable)) and (variable != 'elevation'):
            raise ValueError(f'missing variable `{variable}`')
        kwargs = self._metadata[variable]['latitude']
        return np.arange(kwargs['start'], kwargs['end']+1e-6, kwargs['step'])

    def get_longitude(self, variable):
        if (not self.has_variable(variable)) and (variable != 'elevation'):
            raise ValueError(f'missing variable `{variable}`')
        kwargs = self._metadata[variable]['longitude']
        return np.arange(kwargs['start'], kwargs['end']+1e-6, kwargs['step'])

    def get_elevation(self):
        return self._load_array('elevation')

    def get(self, variable, times=None, lons=None, lats=None, regrid_method='bilinear'):
        if not self.has_variable(variable):
            raise ValueError(f'missing variable `{variable}`')

        data_lats = self.get_latitude(variable)
        data_lons = self.get_longitude(variable)
        data = self._load_array(variable)

        if (lons is None and lats is not None) or (lons is not None and lats is None):
            raise ValueError('lats and lons, or none of them, must be provided')

        if lons is not None and lats is not None:
            # spatial regridding...
            target_lons = np.array(lons, ndmin=1)
            target_lats = np.array(lats, ndmin=1)
            data = regrid(
                data_lons, data_lats, data,
                target_lons, target_lats, method=regrid_method)

            data_lats = target_lats
            data_lons = target_lons

        if times is not None:
            # expand dataset in the temporal dimension to span the target period...
            target_times = np.array(times, dtype='datetime64[ns]')
            years = np.unique(target_times.astype('datetime64[Y]')).astype('i2') + 1970
            expanded_times = np.array(
                [f'{yr}-{mo:02d}-15' for yr in years for mo in range(1, 13)],
                dtype='datetime64[ns]')
            expanded_data = np.vstack([data for _ in range(len(years))])

            # perform the temporal interpolation to the target times...
            kwargs = dict(kind=2, fill_value='extrapolate')
            xi = self._get_fractional_year(expanded_times)
            x = self._get_fractional_year(target_times)
            data = interp1d(xi, expanded_data, axis=0, **kwargs)(x)

        if np.isscalar(lons) and np.isscalar(lats):
            return data[:, 0]
        return data

    def get_atmos(self, times=None, lons=None, lats=None, variables=None, regrid_method='bilinear'):
        req_variables = self.variables if variables is None else variables
        with cf.ThreadPoolExecutor(max_workers=len(req_variables)) as executor:

            futures = {executor.submit(self.get, variable, times, lons, lats, regrid_method): variable
                       for variable in req_variables}
            logger.debug('futures submitted!!')

            data = {}
            for future in cf.as_completed(futures):
                variable = futures[future]
                logger.debug(f'variable `{variable}` completed')
                try:
                    data[variable] = future.result()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error(f'the thread for variable `{variable}` generated and exception: {exc}')

            return data

    def _get_fractional_year(self, times):
        one_day = np.timedelta64(1, 'D')
        one_year = np.timedelta64(1, 'Y')
        jan_1st = times.astype('datetime64[Y]').astype('datetime64[D]')
        dec_31st = jan_1st.astype('datetime64[Y]') + one_year - one_day
        year_length = dec_31st - jan_1st + one_day
        year_fraction = (times - jan_1st) / year_length
        return times.astype('datetime64[Y]').astype('f4') + 1970 + year_fraction

    def _load_array(self, variable):
        if (not self.has_variable(variable)) and (variable != 'elevation'):
            raise ValueError(f'missing variable `{variable}`')

        file_name = self._path / f'{variable}.bl2'
        data = blosc2.load_array(file_name.as_posix())

        visible_modules = sys.modules[__name__]
        visible_class_names, visible_classes = zip(*inspect.getmembers(visible_modules, inspect.isclass))
        for filter_descr in self._metadata[variable]['filters']:
            filter_cls_name = filter_descr['class']
            filter_cls_kwargs = filter_descr['kwargs']
            if filter_cls_name not in visible_class_names:
                raise ValueError(f'unknown filter of type `{filter_descr["class"]}`')
            filter_cls = visible_classes[visible_class_names.index(filter_cls_name)]
            cfilter = filter_cls(**filter_cls_kwargs)
            data = cfilter.decode(data)

        return data


def create_daily_dataset(year, target_path='daily/{year}'):
    VARIABLES = [('albedo', 3), ('pressure', 0), ('ozone', 3), ('pwater', 2),
                 ('alpha', 2), ('beta', 3), ('ssa', 3), ('elevation', 0)]

    def get_target_path(**kwargs):
        return target_path.format(**kwargs)

    def get_source_file_name(**kwargs):
        db_root = Path('/home/jararias/.solarpandas-data/merra2_daily')
        if kwargs['variable'] == 'elevation':
            return db_root / 'merra2_elevation.nc4'
        return db_root / '{variable}/merra2_{variable}_daily_time_chunked_{year}.nc4'.format(**kwargs)

    for variable, precision in VARIABLES:
        logger.info(f'Processing variable `{variable}` / year {year}')

        path = get_target_path(year=year)

        file_name = get_source_file_name(variable=variable, year=year)
        if not file_name.exists():
            logger.warning(f'missing file `{file_name}`. Skipping')

        values, lon, lat, times = read_netcdf(file_name, variable)

        if variable == 'elevation':
            save_array(path, variable, values=values[0], precision=precision)
            update_metadata(path, variable, latitude=lat, longitude=lon)
            continue

        time_start = np.datetime64(times[0], 'ns')
        time_end = np.datetime64(times[-1], 'ns')

        # CREATE A 3-DAYS TIME HALO (where possible)...
        #   this halo guarantees that times can be extracted from the very beginning
        #   and very end of the year without using extrapolation while keeping the
        #   time series continuity in the transition from one year to the following

        # previous file...
        file_name = get_source_file_name(variable=variable, year=year-1)
        if file_name.exists():
            prev_values, _, _, times = read_netcdf(file_name, variable)
            values = np.r_[prev_values[-3:, ...], values]
            time_start = np.datetime64(times[-3], 'ns')

        # next file...
        file_name = get_source_file_name(variable=variable, year=year+1)
        if file_name.exists():
            next_values, _, _, times = read_netcdf(file_name, variable)
            values = np.r_[values, next_values[:3, ...]]
            time_end = np.datetime64(times[2], 'ns')

        save_array(path, variable, values=values, precision=precision)
        times = {'start': str(time_start), 'end': str(time_end), 'delta': [1, 'D']}
        update_metadata(path, variable, latitude=lat, longitude=lon, times=times)


class DailyDataset:
    def __init__(self, path):
        self._path = Path(path)
        metadata_file_name = self._path / 'metadata.json'
        if not metadata_file_name.exists():
            raise ValueError(f'missing required file `{metadata_file_name}`')
        self._metadata = json.load(metadata_file_name.open())

        self._variables = [key for key, value in self._metadata.items()
                           if key != 'elevation' and isinstance(value, dict)]

        year = []
        one_ns = np.timedelta64(1, 'ns')
        for variable in self._variables:
            start = self._metadata[variable]['times']['start']
            stop = self._metadata[variable]['times']['end']
            step = np.timedelta64(*self._metadata[variable]['times']['delta'])
            times = np.arange(start, np.datetime64(stop) + one_ns, step)
            # detect the year that corresponds to this dataset...
            years, counts = np.unique(times.astype('datetime64[Y]'), return_counts=True)
            year.append(years[np.argmax(counts)])

        try:
            self._year = np.unique(year).item().year
        except ValueError as exc:
            raise ValueError(f'dataset spanning multiple years: {np.unique(year)}') from exc

    @property
    def variables(self):
        return self._variables

    @property
    def year(self):
        return self._year

    def has_variable(self, variable):
        return variable in self.variables

    def get_latitude(self, variable):
        if (not self.has_variable(variable)) and (variable != 'elevation'):
            raise ValueError(f'missing variable `{variable}`')
        kwargs = self._metadata[variable]['latitude']
        return np.arange(kwargs['start'], kwargs['end']+1e-6, kwargs['step'])

    def get_longitude(self, variable):
        if not self.has_variable(variable) and (variable != 'elevation'):
            raise ValueError(f'missing variable `{variable}`')
        kwargs = self._metadata[variable]['longitude']
        return np.arange(kwargs['start'], kwargs['end']+1e-6, kwargs['step'])

    def get_times(self, variable):
        if not self.has_variable(variable):
            raise ValueError(f'missing variable `{variable}`')
        one_ns = np.timedelta64(1, 'ns')
        start = self._metadata[variable]['times']['start']
        stop = self._metadata[variable]['times']['end']
        step = np.timedelta64(*self._metadata[variable]['times']['delta'])
        return np.arange(start, np.datetime64(stop) + one_ns, step)

    def get_elevation(self):
        return self._load_array('elevation')

    def get(self, variable, times=None, lons=None, lats=None, regrid_kwargs=None, interp_kwargs=None):
        if not self.has_variable(variable):
            raise ValueError(f'missing variable `{variable}`')

        logger.debug(f'reading data on variable `{variable}`')
        data_times = self.get_times(variable)
        data_lons = self.get_longitude(variable)
        data_lats = self.get_latitude(variable)
        data = self._load_array(variable)

        # ...and set the time limits for the dataset (even if it has a halo)
        data_time_min = np.datetime64(self.year-1970, 'Y').astype('datetime64[ns]')
        data_time_max = np.datetime64(self.year-1970 + 1, 'Y').astype('datetime64[ns]')
        logger.debug(f'dataset year: {self.year}  min. time: '
                     f'{data_time_min}  max. time: {data_time_max}')

        if (lons is None and lats is not None) or (lons is not None and lats is None):
            raise ValueError('lats and lons, or none of them, must be provided')

        if lons is not None and lats is not None:
            logger.debug(f'regridding on variable `{variable}`')
            # spatial regridding...
            kwargs = {'method': 'bilinear'}
            kwargs.update(regrid_kwargs or {})
            target_lons = np.array(lons, ndmin=1)
            target_lats = np.array(lats, ndmin=1)
            data = regrid(data_lons, data_lats, data, target_lons, target_lats, **kwargs)

            data_lats = target_lats
            data_lons = target_lons

        if times is not None:
            # perform the temporal interpolation to the target times...
            logger.debug(f'temporal interpolation on variable `{variable}`')
            # the variables that changes smoothly throughout time are interpolated
            # linearly to save time. If there are nan values in any array, use linear
            # interpolation because quadratic and cubic then fill all with nans
            kind = {'beta': 2, 'alpha': 2, 'pwater': 2}.get(variable, 1)
            if np.any(np.isnan(data)):
                kind = 1
            kwargs = {'kind': kind, 'fill_value': np.nan}
            kwargs.update(interp_kwargs or {})
            if interp_kwargs:
                logger.debug(f'{kwargs=}')

            target_times = np.array(times, dtype='datetime64[ns]')

            xi = data_times.astype('f8')
            x = target_times.astype('f8')
            data = interp1d(xi, data, axis=0, **kwargs)(x)

            mask = (target_times < data_time_min) | (target_times >= data_time_max)
            data[mask] = np.nan
            data_times = target_times

        if np.isscalar(lons) and np.isscalar(lats):
            logger.debug(f'"{variable}".shape={data[:, 0].shape}')
            return data[:, 0]

        logger.debug(f'"{variable}".shape={data.shape}')
        return data

    def get_atmos(self, **kwargs):
        with cf.ThreadPoolExecutor(max_workers=len(self.variables)) as executor:

            futures = {executor.submit(self.get, variable, **kwargs): variable
                       for variable in self.variables}
            logger.debug('futures submitted!!')

            data = {}
            for future in cf.as_completed(futures):
                variable = futures[future]
                logger.debug(f'variable `{variable}` completed')
                try:
                    data[variable] = future.result()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error(f'the thread for variable `{variable}` generated and exception: {exc}')

            return data

    def _load_array(self, variable):
        if (not self.has_variable(variable)) and (variable != 'elevation'):
            raise ValueError(f'missing variable `{variable}`')

        file_name = self._path / f'{variable}.bl2'
        data = blosc2.load_array(file_name.as_posix())

        visible_modules = sys.modules[__name__]
        visible_class_names, visible_classes = zip(*inspect.getmembers(visible_modules, inspect.isclass))
        for filter_descr in self._metadata[variable]['filters']:
            filter_cls_name = filter_descr['class']
            filter_cls_kwargs = filter_descr['kwargs']
            if filter_cls_name not in visible_class_names:
                raise ValueError(f'unknown filter of type `{filter_descr["class"]}`')
            filter_cls = visible_classes[visible_class_names.index(filter_cls_name)]
            cfilter = filter_cls(**filter_cls_kwargs)
            data = cfilter.decode(data)

        return data


class DailyDatasets:
    def __init__(self, path):
        root = Path(path)
        relpaths = [p for p in root.iterdir() if p.is_dir()]
        self._datasets = sorted([DailyDataset(p) for p in relpaths], key=lambda dd: dd.year)

    def __repr__(self):
        return "[" + ', '.join([f"DailyDataset@{dd._path}" for dd in self._datasets]) + "]"

    def _iter_times(self, times):
        for dd in self._datasets:
            lower_bound = np.datetime64(dd.year-1970, 'Y').astype('datetime64[ns]')
            upper_bound = np.datetime64(dd.year+1-1970, 'Y').astype('datetime64[ns]')
            yield dd, (lower_bound <= times) & (times < upper_bound)

    def get_elevation(self):
        return self._datasets[0].get_elevation()

    def get(self, variable, times=None, lons=None, lats=None, regrid_kwargs=None, interp_kwargs=None):
        kwargs = dict(lons=lons, lats=lats, regrid_kwargs=regrid_kwargs, interp_kwargs=interp_kwargs)
        if times is None:
            return np.concatenate([dd.get(variable, times, **kwargs) for dd in self._datasets], axis=0)
        return np.concatenate([dd.get(variable, times[domain], **kwargs)
                               for dd, domain in self._iter_times(times)], axis=0)

    def get_atmos(self, times=None, lons=None, lats=None, regrid_kwargs=None, interp_kwargs=None):
        kwargs = dict(lons=lons, lats=lats, regrid_kwargs=regrid_kwargs, interp_kwargs=interp_kwargs)

        atmos = defaultdict(list)
        if times is None:
            for dd in self._datasets:
                for variable, values in dd.get_atmos(**({'times': times} | kwargs)).items():
                    atmos[variable].append(values)
        else:
            for dd, domain in self._iter_times(times):
                for variable, values in dd.get_atmos(**({'times': times[domain]} | kwargs)).items():
                    atmos[variable].append(values)

        return {variable: np.concatenate(atmos[variable], axis=0) for variable in atmos}
