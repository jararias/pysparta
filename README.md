![python versions](https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10-blue.svg)

# CSRAD

**csrad** is a Python package for clear-sky broadband solar irradiance assessment with the *Solar PArameterization of the Radiative Transfer of the Atmosphere* (SPARTA), and optionally with other models (see [csrad_public_models](https://gitlab.solargis.com/backend/csrad_public_models)).

### Main features

SPARTA, and any optional clear-sky solar irradiance model within *csrad*, compute the broadband global, direct and diffuse clear-sky solar irradiances at the surface. In particular, all models provide the output in a dictionary which has the following mandatory keys:

- *dni*, broadband direct normal irradiance, in W/m<sup>2</sup>,
- *dhi*, broadband direct horizontal irradiance, in W/m<sup>2</sup>,
- *dif*, broadband diffuse horizontal irradiance, in W/m<sup>2</sup>,
- *ghi*, broadband global horizontal irradiance, in W/m<sup>2</sup>

In addition, SPARTA provides the circumsolar irradiance (*csi*). Some models may provide other outputs, such as REST2, which also computes illuminances and photosynthetically active (PAR) fluxes. See the inline documentation in each model for further details using, for instance,
```python3 -m pydoc csrad.csmodels```

### Installation notes

*csrad* is available in *nexus.solargis.com* for Linux x86_64 machines with Python>=3.8. (For Python==2.7, see [csrad2](https://gitlab.solargis.com/backend/csrad2)). Having access to the *nexus*'s Python repo, the installation should be as easy as:

```python3 -m pip install csrad```

or

```python3 -m pip install csrad[with_public_models]```

to also install extra "public" clear-sky solar irradiance models.

##### If the installation fails:

1. If the error raises the following message at import time:

   ```
   RuntimeError: module compiled against API version <whatever> but this version of numpy is <whatever>
   Traceback (most recent call last):
     ...
   ImportError: numpy.core.multiarray failed to import
   ```
  
  You can upgrade `numpy` (`python3 -m pip install -U numpy`) or you can try cloning to local and "pip install" from the cloned sources including the flag ```--no-build-isolation```. Be aware that you will need a fortran compiler in your system. If you are in a Ubuntu machine, you can install the GNU's fortran with ```apt-get install gfortran```. To learn more about this error, see [this](https://discuss.python.org/t/pep-517-how-to-pick-compatible-build-dependencies/2460) enlightening discussion in the context of PEP517.

2. Double check that you are providing the proper pip index. There are several means to do that:

   - In the call to pip install:
     ```python3 -m pip install --index-url=https://nexus.solargis.com/repository/pypi/simple csrad```
   - Using an environment variable:
     ```export PIP_INDEX_URL="https://nexus.solargis.com/repository/pypi/simple"```
   - In the file ~/.config/pip/pip.conf, with the following settings:
     ```
     [global]
     default-timeout = 10
     respect-virtualenv = true
     extra-index-url = https://nexus.solargis.com/repository/pypi/simple
     ```

### Usage guidelines

The available models in *csrad* are inspected as follows:

```python
import csrad
print(csrad.AVAILABLE_MODELS)
```

They all are functions whose input parameters include:
- some for **solar position** (cosine of solar zenith angle, *cosz*, and eccentricity correction factor, *ecf*, to account for the actual distance between sun and earth),
- some for **atmospheric conditions** (pressure, total column water vapor content, aerosols...),
- optionally, some models also have other options that allow configuring the **operation of the model**. For instance, SPARTA has an option to set the half field of view angle to compute circumsolar irradiance.

All the input parameters are set to default values. Detailed information can be consulted in the inline documentation.

##### Example 1
Simulate the clear-sky surface solar irradiance every 15 minutes during 2018-10-01. We know that, for that day, the Angstrom turbidity coefficient, _beta_, is 0.3. Then:

```python
import sunpos
import csrad
import pandas as pd

times = pd.date_range('2018-10-01T00:00:00', periods=24*4, freq='15T')
sp = sunpos.sites(times, latitude=37.5, longitude=-3.5)
res = csrad.SPARTA(cosz=sp.cosz, ecf=sp.ecf, beta=0.3)
```

*res* is a dictionary that holds the simulation results. Because *cosz* and *ecf* are time series, the outputs are so too. In particular, for example `res['dni'].shape` is the same as `times.shape`. All the remaining inputs to ```SPARTA(...)``` are defaulted.

> **_NOTE:_** see https://gitlab.solargis.com/backend/sunpos for details about *sunpos*

Note also that in the example, *beta* is 0.3, i.e., it is an scalar whereas the other input arguments are time series. The code _automagically_ expands the dimensions of *beta* to match the shape of the input arguments. This also occurs with the rest of parameters, which are left to their default values. In the most general case, *beta* and the other input arguments must have all the same shape and their values can change in the temporal and spatial dimensions. As a particular case, when one (or various) input arguments are scalars (as in the example above), *csrad* tries to expand them so that all the inputs have the same shape.

Following with the previous simulation, to plot *ghi* we could do the following:

```python
import pylab as pl
pl.plot(times, res['ghi'], 'r-')
pl.show()
```

##### Example 2

*csrad* can also deal with spatio-temporal grids. For example, we can do:

```python
import sunpos
import csrad
import pandas as pd

times = pd.date_range('2018-10-01T00:00:00', periods=24, freq='60T')
# lat-lon grid with half-degree resolution
lats = np.arange(-90, 90, 0.5)
lons = np.arange(-180, 180, 0.5)
sp = sunpos.regular_grid(times, latitude=lats, longitude=lons)
cosz = sp.cosz  # is a 3D numpy array with shape (len(times), len(lats), len(lons))
ecf = sp.ecf  # is a 1D numpy array with shape (len(times),)
# we must reshape ecf to match the 3D shape of cosz.
# csrad cannot do it automatically because ecf is not an scalar
ecf = ecf[:, None, None] * np.ones(cosz.shape)
res = csrad.SPARTA(cosz=cosz, ecf=ecf, beta=0.3, n_cpus=2)
```
*beta* and the rest of scalar input arguments are expanded to match the shape of cosz and ecf. In this example, `res['ghi'].shape` is `(len(times), len(lats), len(lons))`.
