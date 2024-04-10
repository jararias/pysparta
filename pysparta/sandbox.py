
import time
from collections import OrderedDict as ODict

import numpy as np


def assert_that(cond, msg='', exc=None):
    if not cond:
        raise (exc or AttributeError)(msg)


def random_dataset(n_points=50, cosz=(0., 1.), ozone=(0.0, 0.6),
                   pressure=(300., 1100.), pwater=(0.0, 10.),
                   albedo=(0.0, 1.0), beta=(0.0, 1.2), alpha=(0.0, 2.5),
                   ssa=(0.5, 1.0), asy=(0.5, 1.0), seed=None):

    if seed is None:
        seed = int(1e9 * (time.time() % 1))
    np.random.seed(seed)

    random_values = ODict()
    random_values['cosz'] = np.random.uniform(*cosz, size=n_points)
    random_values['ozone'] = np.random.uniform(*ozone, size=n_points)
    random_values['pressure'] = np.random.uniform(*pressure, size=n_points)
    random_values['pwater'] = np.random.uniform(*pwater, size=n_points)
    random_values['albedo'] = np.random.uniform(*albedo, size=n_points)
    random_values['beta'] = np.random.uniform(*beta, size=n_points)
    random_values['alpha'] = np.random.uniform(*alpha, size=n_points)
    random_values['ssa'] = np.random.uniform(*ssa, size=n_points)
    random_values['asy'] = np.random.uniform(*asy, size=n_points)

    return random_values
