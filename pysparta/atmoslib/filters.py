
import numpy as np


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
