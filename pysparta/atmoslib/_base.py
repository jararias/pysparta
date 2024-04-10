
import abc
import json
from pathlib import Path

import numpy as np

from .sandbox import assert_that


class BaseAtmosphere(metaclass=abc.ABCMeta):

    def __init__(self):
        assert_that(
            self.database_path.exists(),
            f'missing path `{self.database_path}`')

        metadata_path = self.database_path / 'metadata.json'
        assert_that(
            metadata_path.exists(),
            f'missing metadata file `{metadata_path}`')

        self._metadata = json.load(metadata_path.open())

        assert_that(
            'elevation' in self._metadata,
            'missing `elevation` in metadata file')

        self._variables = [
            key for key, value in self._metadata.items()
            if key != 'elevation' and isinstance(value, dict)]

    def __init_subclass__(cls, database_path, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.database_path = Path(database_path)

    @property
    def variables(self):
        """list of variables included in this atmosphere"""
        return self._variables

    def has_variable(self, variable):
        """if variable is included in this atmosphere"""
        return variable in self.variables

    def get_latitudes(self, variable):
        """1D grid of latitudes of variable"""
        assert_that(
            variable in self._metadata,
            f'missing variable `{variable}`')
        kwargs = self._metadata[variable]['latitude']
        return np.arange(kwargs['start'], kwargs['end']+1e-6, kwargs['step'])

    def get_longitudes(self, variable):
        """1D grid of longitudes of variable"""
        assert_that(
            variable in self._metadata,
            f'missing variable `{variable}`')
        kwargs = self._metadata[variable]['longitude']
        return np.arange(kwargs['start'], kwargs['end']+1e-6, kwargs['step'])

    def get_elevation(self):
        """2D grid of elevations"""
        return self._load_array('elevation')

    @abc.abstractmethod
    def get_variable(self, variable, times=None, sites=None, regular_grid=None,
                     space_interp=None, time_interp=None):
        pass

    @abc.abstractmethod
    def get_atmosphere(self, times=None, sites=None, regular_grid=None,
                       variables=None, space_interp=None, time_interp=None):
        pass

    @abc.abstractmethod
    def _load_array(self, variable):
        pass
