
import re
import time
from io import StringIO
from pathlib import Path
from copy import deepcopy
from functools import reduce
from datetime import datetime, timezone
import concurrent.futures as cf

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from siphon.catalog import TDSCatalog
from siphon.ncss import ResponseRegistry


logger.disable(__name__)
logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)


def download_dataset(dataset, lon, lat, variables):
    ncss = dataset.subset()
    time_span = ncss.metadata.time_span
    ds_start = pd.Timestamp(time_span['begin']).to_pydatetime()
    ds_end = pd.Timestamp(time_span['end']).to_pydatetime()

    query = ncss.query()
    query.time_range(ds_start, ds_end)
    query.lonlat_point(lon, lat)
    query.variables(*variables)
    query.accept('csv')

    resp = ncss.get_query(query)
    response_handlers = ResponseRegistry()
    data_str = response_handlers(resp, ncss.unit_handler).decode('utf-8')
    return pd.read_csv(StringIO(data_str), parse_dates=['time'], index_col=0)


def download_monthly_variables(date_start, date_end, lon, lat, variables, catalog_pattern):

    MAX_WORKERS = 5
    SERVER_ROOT = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/thredds/'

    days = pd.date_range(date_start, date_end, freq='T')
    year_and_month = pd.to_datetime(np.array(days, dtype='datetime64[M]')).unique()

    df = pd.DataFrame()

    kwargs = dict(lon=lon, lat=lat, variables=variables)

    progress_bar_msg = 'Monthly '
    if len(variables) == 1:
        progress_bar_msg += variables[0]
    elif len(variables) == 2:
        progress_bar_msg += ' and '.join(variables)
    else:
        progress_bar_msg += ', '.join(variables[:-1]) + f' and {variables[-1]}'

    tqdm_kwargs = dict(desc=progress_bar_msg, total=len(year_and_month))
    tqdm_it = 0

    def select_dataset(cds, ym):
        ym_stamp = f'{ym.year}{ym.month:02d}'
        ds_name = list(filter(lambda ds: ym_stamp in ds, cds.datasets))[0]
        return cds.datasets[ds_name]

    for year in sorted(year_and_month.year.unique()):

        url = SERVER_ROOT + catalog_pattern.format(year=year)

        catalog = TDSCatalog(url)
        logger.debug(catalog.catalog_url)

        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            futures = {}
            for ym in year_and_month[year_and_month.year == year]:
                ds = select_dataset(catalog, ym)
                futures[executor.submit(download_dataset, ds, **kwargs)] = ds

            logger.debug('  futures submitted!')

            for future in tqdm(cf.as_completed(futures), initial=tqdm_it, **tqdm_kwargs):
                ds = futures[future]
                try:
                    data = future.result()
                    gridcell_lat = data.pop('latitude[unit="degrees_north"]').unique().item()
                    gridcell_lon = data.pop('longitude[unit="degrees_east"]').unique().item()
                    df = pd.concat([df, data], axis=0)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.debug(f'{ds} generated and exception: {exc}')

        tqdm_it += len(futures)

    df = df.sort_index()
    df.index = df.index.tz_convert(None)
    return df, gridcell_lon, gridcell_lat


def download_monthly_aerosol_optical_properties(date_start, date_end, lon, lat, variables):
    return download_monthly_variables(
        date_start, date_end, lon, lat, variables,
        catalog_pattern='catalog/M2TMNXAER.5.12.4/{year}/catalog.xml')


def download_monthly_single_level_variables(date_start, date_end, lon, lat, variables):
    return download_monthly_variables(
        date_start, date_end, lon, lat, variables,
        catalog_pattern='catalog/M2TMNXSLV.5.12.4/{year}/catalog.xml')


def download_monthly_radiation_diagnostics(date_start, date_end, lon, lat, variables):
    return download_monthly_variables(
        date_start, date_end, lon, lat, variables,
        catalog_pattern='catalog/M2TMNXRAD.5.12.4/{year}/catalog.xml')


def download_hourly_variables(date_start, date_end, lon, lat, variables, catalog_pattern):

    MAX_WORKERS = 5  # it appears that 5 is a max for allowed concurrent connections !!
    SERVER_ROOT = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/thredds/'

    days = pd.date_range(date_start, date_end, freq='T')
    n_days = len(pd.to_datetime(np.array(days, dtype='datetime64[D]')).unique())
    year_and_month = pd.to_datetime(np.array(days, dtype='datetime64[M]')).unique()

    df = pd.DataFrame()

    kwargs = dict(lon=lon, lat=lat, variables=variables)

    progress_bar_msg = 'Hourly '
    if len(variables) == 1:
        progress_bar_msg += variables[0]
    elif len(variables) == 2:
        progress_bar_msg += ' and '.join(variables)
    else:
        progress_bar_msg += ', '.join(variables[:-1]) + f' and {variables[-1]}'

    tqdm_kwargs = dict(desc=progress_bar_msg, total=n_days)
    tqdm_it = 0

    for ym in year_and_month:

        url = SERVER_ROOT + catalog_pattern.format(year=ym.year, month=ym.month)

        catalog = TDSCatalog(url)
        logger.debug(catalog.catalog_url)

        # filter datasets to keep only the requested ones...
        # I use deepcopy + pop to ensure that datasets is the same class as catalog.datasets
        def get_date(ds_name):
            return datetime.strptime(Path(ds_name).stem.split('.')[-1], '%Y%m%d').date()

        datasets = deepcopy(catalog.datasets)
        date_s, date_e = date_start.date(), date_end.date()
        for ds in [ds for ds in datasets if not date_s <= get_date(ds) <= date_e]:
            datasets.pop(ds)

        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            futures = {executor.submit(download_dataset, ds, **kwargs): ds
                       for ds in datasets.values()}

            logger.debug('  futures submitted!')

            for future in tqdm(cf.as_completed(futures), initial=tqdm_it, **tqdm_kwargs):
                ds = futures[future]
                try:
                    data = future.result()
                    gridcell_lat = data.pop('latitude[unit="degrees_north"]').unique().item()
                    gridcell_lon = data.pop('longitude[unit="degrees_east"]').unique().item()
                    df = pd.concat([df, data], axis=0)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.debug(f'{ds} generated and exception: {exc}')

        tqdm_it += len(datasets)

    df = df.sort_index()
    df.index = df.index.tz_convert(None)
    return df, gridcell_lon, gridcell_lat


def download_hourly_aerosol_optical_properties(date_start, date_end, lon, lat, variables):
    return download_hourly_variables(
        date_start, date_end, lon, lat, variables,
        catalog_pattern='catalog/M2T1NXAER.5.12.4/{year}/{month:02d}/catalog.xml')


def download_hourly_single_level_variables(date_start, date_end, lon, lat, variables):
    return download_hourly_variables(
        date_start, date_end, lon, lat, variables,
        catalog_pattern='catalog/M2T1NXSLV.5.12.4/{year}/{month:02d}/catalog.xml')


def download_hourly_radiation_diagnostics(date_start, date_end, lon, lat, variables):
    return download_hourly_variables(
        date_start, date_end, lon, lat, variables,
        catalog_pattern='catalog/M2T1NXRAD.5.12.4/{year}/{month:02d}/catalog.xml')


def get_clearsky_atmosphere(date_start, date_end, lon, lat, hourly=None, monthly=None):
    """
    hourly must be None or list of valid_variable_names (idem for monthly)
    """

    t_start = time.perf_counter()

    # PERFECT CAPTURE OF HOURLY + MONTHLY CHOICES...

    VALID_VARIABLE_NAMES = ('alpha', 'beta', 'ssa', 'pressure', 'pwater', 'ozone', 'albedo')

    if not isinstance(hourly, (list, type(None))):
        raise ValueError('`hourly` must be None or list')

    if hourly is not None:
        if len(not_valid := [name for name in hourly if name not in VALID_VARIABLE_NAMES]):
            raise ValueError(f'found unknown variable names `hourly`: {not_valid}')

    if not isinstance(monthly, (list, type(None))):
        raise ValueError('`monthly` must be None or list')

    if monthly is not None:
        if len(not_valid := [name for name in monthly if name not in VALID_VARIABLE_NAMES]):
            raise ValueError(f'found unknown variable names in `monthly`: {not_valid}')

    # default configuration...
    hourly_variables = {'alpha', 'beta', 'pwater'}
    monthly_variables = {'ozone', 'albedo', 'pressure', 'ssa'}

    if hourly is not None:
        hourly_variables = hourly_variables.union(hourly)
        monthly_variables = monthly_variables.difference(hourly_variables)

    if monthly is not None:
        monthly_variables = monthly_variables.union(monthly)
        hourly_variables = hourly_variables.difference(monthly_variables)

    assert hourly_variables.union(monthly_variables) == set(VALID_VARIABLE_NAMES)

    logger.debug(f'{hourly_variables=}')
    logger.debug(f'{monthly_variables=}')

    # CHECK INPUT DATATIMES, MOVE TO UTC, AND CONVERT TO NAIVE DATETIMES

    date_start = pd.to_datetime(date_start).to_pydatetime()
    if date_start.tzinfo is not None:
        date_start = date_start.astimezone(timezone.utc).replace(tzinfo=None)

    date_end = pd.to_datetime(date_end).to_pydatetime()
    if date_end.tzinfo is not None:
        date_end = date_end.astimezone(timezone.utc).replace(tzinfo=None)

    # CHECK LON & LAT

    if not -90 < lat < 90:
        raise ValueError(f'{lat=} out of bounds')

    if not -180 <= lon < 180:
        raise ValueError(f'{lon=} out of bounds')

    # BUILD TIME GRID AND CREATE TARGET DATAFRAME

    date_s = date_start.replace(hour=0, minute=30, second=0)
    date_e = date_end.replace(hour=23, minute=30, second=0)
    time_grid = pd.date_range(date_s, date_e, freq='H')
    df = pd.DataFrame(index=time_grid)

    # AEROSOL OPTICAL PROPERTIES...

    name_map = {'alpha': ['TOTANGSTR'], 'beta': ['TOTEXTTAU', 'TOTANGSTR'], 'ssa': ['TOTEXTTAU', 'TOTSCATAU']}

    # hourly requests !!
    if (requested_variables := {'alpha', 'beta', 'ssa'}.intersection(hourly_variables)):
        logger.info(f'Hourly aerosols requested: {requested_variables}')

        variables = [name_map[name] for name in requested_variables]
        variables = list(set(reduce(lambda a, b: a + b, variables)))  # flatten and remove duplicates

        df_aer, gridcell_lon, gridcell_lat = download_hourly_aerosol_optical_properties(
            date_start=date_start, date_end=date_end, lon=lon, lat=lat, variables=variables)

        df_aer.rename(columns=lambda name: name.rstrip('[unit="1"]'), inplace=True)

        if 'alpha' in requested_variables:
            df_aer = df_aer.eval("""alpha = TOTANGSTR""")

        if 'beta' in requested_variables:
            df_aer = df_aer.eval("""beta = TOTEXTTAU*(0.55**TOTANGSTR)""")

        if 'ssa' in requested_variables:
            df_aer = df_aer.eval("""ssa = TOTSCATAU / TOTEXTTAU""")

        df_aer.drop(columns=['TOTANGSTR', 'TOTEXTTAU', 'TOTSCATAU'], errors='ignore', inplace=True)
        df = pd.concat([df, df_aer], axis=1)

    # monthly requests !!
    if (requested_variables := {'alpha', 'beta', 'ssa'}.intersection(monthly_variables)):
        logger.info(f'Monthly aerosols requested: {requested_variables}')

        variables = [name_map[name] for name in requested_variables]
        variables = list(set(reduce(lambda a, b: a + b, variables)))  # flatten and remove duplicates

        # increase date_end in one month to prevent extrapolation later
        if date_end.month < 12:
            date_end_ = date_end.replace(month=date_end.month+1)
        else:
            date_end_ = date_end.replace(year=date_end.year+1, month=1)

        df_aer, gridcell_lon, gridcell_lat = download_monthly_aerosol_optical_properties(
            date_start=date_start, date_end=date_end_, lon=lon, lat=lat, variables=variables)

        df_aer.rename(columns=lambda name: name.rstrip('[unit="1"]'), inplace=True)

        if 'alpha' in requested_variables:
            df_aer = df_aer.eval("""alpha = TOTANGSTR""")

        if 'beta' in requested_variables:
            df_aer = df_aer.eval("""beta = TOTEXTTAU*(0.55**TOTANGSTR)""")

        if 'ssa' in requested_variables:
            df_aer = df_aer.eval("""ssa = TOTSCATAU / TOTEXTTAU""")

        df_aer.drop(columns=['TOTANGSTR', 'TOTEXTTAU', 'TOTSCATAU'], errors='ignore', inplace=True)
        full_hourly_times = pd.date_range(df_aer.index[0], df_aer.index[-1], freq='H')
        df_aer = df_aer.reindex(full_hourly_times).interpolate(method='linear').reindex(df.index)
        df = pd.concat([df, df_aer], axis=1)

    # SINGLE LEVEL VARIABLES...

    name_map = {'pressure': ['PS'], 'ozone': ['TO3'], 'pwater': ['TQV']}

    # hourly requests !!
    if (requested_variables := {'pressure', 'ozone', 'pwater'}.intersection(hourly_variables)):
        logger.info(f'Hourly single-level variables requestes: {requested_variables}')

        variables = [name_map[name] for name in requested_variables]
        variables = list(set(reduce(lambda a, b: a + b, variables)))  # flatten and remove duplicates

        df_slv, gridcell_lon, gridcell_lat = download_hourly_single_level_variables(
            date_start=date_start, date_end=date_end, lon=lon, lat=lat, variables=variables)

        # remove the trailing units from the retrieved variable names...
        df_slv.rename(columns=lambda name: re.match(r"(.*)(\[.*\])", name).groups()[0] or name, inplace=True)

        if 'pressure' in requested_variables:
            df_slv = df_slv.eval("""pressure = PS / 100""")  # hPa

        if 'ozone' in requested_variables:
            df_slv = df_slv.eval("""ozone = TO3 / 1000""")  # atm-cm

        if 'pwater' in requested_variables:
            df_slv = df_slv.eval("""pwater = TQV / 10""")  # atm-cm

        df_slv.drop(columns=['PS', 'TO3', 'TQV'], errors='ignore', inplace=True)
        df = pd.concat([df, df_slv], axis=1)

    # monthly requests !!
    if (requested_variables := {'pressure', 'ozone', 'pwater'}.intersection(monthly_variables)):
        logger.info(f'Monthly single-level variables requested: {requested_variables}')

        variables = [name_map[name] for name in requested_variables]
        variables = list(set(reduce(lambda a, b: a + b, variables)))  # flatten and remove duplicates

        # increase date_end in one month to prevent extrapolation later
        if date_end.month < 12:
            date_end_ = date_end.replace(month=date_end.month+1)
        else:
            date_end_ = date_end.replace(year=date_end.year+1, month=1)

        df_slv, gridcell_lon, gridcell_lat = download_monthly_single_level_variables(
            date_start=date_start, date_end=date_end_, lon=lon, lat=lat, variables=variables)

        df_slv.rename(columns=lambda name: re.match(r"(.*)(\[.*\])", name).groups()[0] or name, inplace=True)

        if 'pressure' in requested_variables:
            df_slv = df_slv.eval("""pressure = PS / 100""")  # hPa

        if 'ozone' in requested_variables:
            df_slv = df_slv.eval("""ozone = TO3 / 1000""")  # atm-cm

        if 'pwater' in requested_variables:
            df_slv = df_slv.eval("""pwater = TQV / 10""")  # atm-cm

        df_slv.drop(columns=['PS', 'TO3', 'TQV'], errors='ignore', inplace=True)
        full_hourly_times = pd.date_range(df_slv.index[0], df_slv.index[-1], freq='H')
        df_slv = df_slv.reindex(full_hourly_times).interpolate(method='linear').reindex(df.index)
        df = pd.concat([df, df_slv], axis=1)

    # GROUND ALBEDO...

    name_map = {'albedo': ['ALBEDO']}

    # hourly requests !!
    if (requested_variables := {'albedo'}.intersection(hourly_variables)):
        logger.info(f'Hourly albedo requested: {requested_variables}')

        variables = [name_map[name] for name in requested_variables]
        variables = list(set(reduce(lambda a, b: a + b, variables)))  # flatten and remove duplicates

        df_rad, gridcell_lon, gridcell_lat = download_hourly_radiation_diagnostics(
            date_start=date_start, date_end=date_end, lon=lon, lat=lat, variables=variables)

        # remove the trailing units from the retrieved variable names...
        df_rad.rename(columns=lambda name: re.match(r"(.*)(\[.*\])", name).groups()[0] or name, inplace=True)

        if 'albedo' in requested_variables:
            df_rad = df_rad.eval("""albedo = ALBEDO""")

        df_rad.drop(columns=['ALBEDO'], errors='ignore', inplace=True)
        df = pd.concat([df, df_rad], axis=1)

    # monthly requests !!
    if (requested_variables := {'albedo'}.intersection(monthly_variables)):
        logger.info(f'Monthly albedo requested: {requested_variables}')

        variables = [name_map[name] for name in requested_variables]
        variables = list(set(reduce(lambda a, b: a + b, variables)))  # flatten and remove duplicates

        # increase date_end in one month to prevent extrapolation later
        if date_end.month < 12:
            date_end_ = date_end.replace(month=date_end.month+1)
        else:
            date_end_ = date_end.replace(year=date_end.year+1, month=1)

        df_rad, gridcell_lon, gridcell_lat = download_monthly_radiation_diagnostics(
            date_start=date_start, date_end=date_end_, lon=lon, lat=lat, variables=variables)

        df_rad.rename(columns=lambda name: re.match(r"(.*)(\[.*\])", name).groups()[0] or name, inplace=True)

        if 'albedo' in requested_variables:
            df_rad = df_rad.eval("""albedo = ALBEDO""")

        df_rad.drop(columns=['ALBEDO'], errors='ignore', inplace=True)
        full_hourly_times = pd.date_range(df_rad.index[0], df_rad.index[-1], freq='H')
        df_rad = df_rad.reindex(full_hourly_times).interpolate(method='linear').reindex(df.index)
        df = pd.concat([df, df_rad], axis=1)

        logger.info(f'Running time: {time.perf_counter() - t_start:.1f} seconds')

    return {'data': df, 'gridcell_lon': gridcell_lon, 'gridcell_lat': gridcell_lat}
