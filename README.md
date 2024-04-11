# SPARTA: *Solar PArameterization of the Radiative Transfer of the Atmosphere*

#### page under construction!

The SPARTA model is fully described (open access) in:

Ruiz-Arias J.A (2023) SPARTA: Solar parameterization for the radiative transfer of the cloudless atmosphere, _Renewable and Sustainable Energy Reviews_, Vol. 188, 113833 doi: [10.1016/j.rser.2023.113833](https://doi.org/10.1016/j.rser.2023.113833)

### Installation

```python
python3 -m pip install git+https://github.com/jararias/pysparta@main
```

### Usage description

`pysparta` can be used out-of-the-box because it is shipped with a long-term (average) clear-sky atmosphere derived from the MERRA-2 atmospheric reanalysis for the period 2010-2021. However, it can be run also from user-provided inputs.

The SPARTA model is accessed as:

```python
from pysparta import SPARTA
```

`SPARTA` is a function with the following signature:

```python
def SPARTA(times=None, sites=None, regular_grid=None, atmos=None, sunwhere_kwargs=None, atmos_kwargs=None, **kwargs):
```

It requires two sets of inputs:

- solar geometry (ultimately, `cosz`, the cosine of the solar zenith angle, and `ecf`, the sun-earth eccentricity correction factor)

- atmospheric parameters (e.g., precipitable water and aerosol Angstrom turbidity), the number of which depend on the model (apart from SPARTA, `pysparta` includes a few more models for benchmarking)

The input arguments of the `SPARTA(...)` function allow providing this information from two different levels: a high one, in which the solar geometry and atmospheric parameters are internally evaluated and retrieved, respectively, and a lower one, in which the parameters are directly provided by the user. Additionally, the two levels can be mixed together (e.g., when a user wants to use parameters from the long-term average atmosphere except for one or some of them that are to be provided externally from another source). The `**kwargs` arguments are used to provide the low-level inputs, while the rest are used to provide the high level inputs, as explained in the following.

- Solar geometry: the user must provide `cosz` and `ecf` as `**kwargs` or must provide the times and locations so that `pysparta` can evaluate them using [sunwhere](https://github.com/jararias/sunwhere). The times (a 1D numpy array of datetime64 objects, or a pandas DatetimeIndex) are provided with the input argument `times`, while the locations are provided either with `sites` or with `regular_grid` (but not both of them at the same time). The two are dictionaries with two mandatory keys, `latitude` and `longitude`, whose values are 1D numpy arrays of floats with the latitudes and longitudes of the target locations. `sites` is intended for calculations over a number of random locations throughout a common time grid (i.e., `times`). `regular_grid` is intended for calculations over a regular (cartesian) grid of locations. `sunwhere_kwargs` can be used to pass arguments to [sunwhere](https://github.com/jararias/sunwhere) if the default configuration is not appropriate (e.g., to select a solar position algorithm other than `psa`, the default one).

- Atmospheric parameters: the user must provide `albedo` (surface albedo), `pressure` (atmospheric pressure, hPa), `ozone` (total-column ozone content, atm-cm), `pwater` (precipitable water, atm-cm), `beta` (aerosol Angstrom turbidity), `alpha` (aerosol Angstrom exponent), `ssa` (aerosol single scattering albedo) and `asy` (aerosol asymmetry parameter) via `**kwargs`, or must select an atmosphere (via `atmos`) to retrieve them. The parameters that are not provided by any means are set to a default value (e.g., `asy` defaults to 0.65).
If the user wants to retrieve the atmospheric parameters from the internal atmosphere, it must provide a valid identifier for the atmosphere in `atmos` (see `pysparta.atmoslib.databases`) and the `times` vector and latitude and longitude vectors, via `sites` or `regular_grid`, as explained above.

If the high-level interface is used, that is, the `times` and `sites` or `regular_grid` inputs are used, the output is a xarray Dataset. Otherwise, the outputs are provided in a dictionary. The output may include some or all of the following variables: `ghi` (global horizonal irradiance), `dhi` (direct horizontal irradiance), `dni` (direct normal irradiance), `dif` (diffuse horizontal irradiance), and `csi` (circumsolar irradiance).

Below, I show some basic usage cases:

```python
from pysparta import SPARTA

times = pd.date_range('2020-01-01', periods=24, freq='1h')
lats = np.random.uniform(-30, 30, 15)
lons = np.random.uniform(-12, 12, 15)
res = SPARTA(times,
             sites={'latitude': lats, 'longitude': lons},
             atmos='merra2_lta')
```

The ouput is a xarray Dataset with dimensions `(time, location)`:

```sh
<xarray.Dataset>
Dimensions:    (time: 24, location: 15)
Coordinates:
  * time       (time) datetime64[ns] 2020-01-01 ... 2020-01-01T23:00:00
  * location   (location) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    longitude  (location) float64 2.144 0.5942 -2.907 -5.5 ... 1.756 1.495 10.4
    latitude   (location) float64 -8.227 -23.73 28.39 ... 8.681 -13.11 -28.06
Data variables:
    dni        (time, location) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
    dhi        (time, location) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
    dif        (time, location) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
    ghi        (time, location) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
    csi        (time, location) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
```

For instance, you can visualize the GHI in the first location using matplotlib as:

```python
res.ghi.isel(location=0).plot()
```

You could also simulate a clean-and-dry atmosfere as follows:

```python
res = SPARTA(times,
             sites={'latitude': lats, 'longitude': lons},
             beta=0.01, pwater=0.1)
```

where all atmospheric parameters are set to their prescribed default values, except `beta` and `pwater` that are nullified (in reality, I use 0.01 and 0.1, respectively, to account for a minimal aerosol and humidity atmospheric content).

Alternativelly, I could have used a long-term average value for all the parameters, other than `beta` and `pwater`, as follows:

```python
res = SPARTA(times,
             sites={'latitude': lats, 'longitude': lons},
             atmos='merra2_lta', beta=0.01, pwater=0.1)
```

(The `beta` and `pwater` values from the `merra2_lta` atmosphere are overwritten to 0.01 and 0.1, respectively).

Furthermore, I could have been used the `merra2_cda` atmosphere, also shipped with SPARTA, and that mimic the former behavior:

```python
res = SPARTA(times,
             sites={'latitude': lats, 'longitude': lons},
             atmos='merra2_lta', beta=0.01, pwater=0.1)
```

If, in any case, I don't want to use the `PSA` solar position algorithm, for some reason, and I prefer the `NREL SPA` instead, I could add

```python
sunwhere_kwargs={'algorithm': 'nrel'}
```

as an additional input argument.

For regular grids, things are pretty similar, just replacing `sites` by `regular_grid`. However, have in mind that now the latitudes and longitudes cannot be at random locations as before, but arranged across a regular grid. For instance:

```python
times = pd.date_range('2020-01-01', periods=24, freq='1h')
lats = np.linspace(-30, 30, 30)
lons = np.linspace(-12, 12, 24)
res = SPARTA(times,
             regular_grid={'latitude': lats, 'longitude': lons},
             atmos='merra2_lta')
```

Now, the dimensions of the output xarray Dataset are different:

```sh
<xarray.Dataset>
Dimensions:    (time: 24, latitude: 30, longitude: 24)
Coordinates:
  * time       (time) datetime64[ns] 2020-01-01 ... 2020-01-01T23:00:00
  * latitude   (latitude) float64 -30.0 -27.93 -25.86 ... 25.86 27.93 30.0
  * longitude  (longitude) float64 -12.0 -10.96 -9.913 ... 9.913 10.96 12.0
Data variables:
    dni        (time, latitude, longitude) float64 0.0 0.0 0.0 ... 0.0 0.0 0.0
    dhi        (time, latitude, longitude) float64 0.0 0.0 0.0 ... 0.0 0.0 0.0
    dif        (time, latitude, longitude) float64 0.0 0.0 0.0 ... 0.0 0.0 0.0
    ghi        (time, latitude, longitude) float64 0.0 0.0 0.0 ... 0.0 0.0 0.0
    csi        (time, latitude, longitude) float64 0.0 0.0 0.0 ... 0.0 0.0 0.0
```

And the GHI output at the 12-th temporal step can be visualized as:

```python
res.ghi.isel(times=12).plot()
```
