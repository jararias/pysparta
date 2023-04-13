
from loguru import logger

from .models.pysparta import SPARTA as py_sparta


logger.disable(__name__)


def SPARTA(cosz=.5, pressure=1013.25, albedo=0.2, pwater=1.4, ozone=0.3,
           beta=0.1, alpha=1.3, ssa=0.92, asy=0.65, ecf=1, csi_param='sparta',
           csi_hfov=2.5, transmittance_scheme='interdependent', engine='numpy'):
    """
    [S]olar [PA]rameterization of the [R]adiative [T]ransfer of the
    [A]tmosphere [SPARTA] A 2-band broadband clear-sky solar radiation model.
    The bands expand the UV-VIS region (280 - 700 nm) and the near IR
    region (700 - 4000 nm)
    Parameters
    ----------
      cosz  : array-like, optional
          Cosine of solar zenith angle.
      pressure : array-like, optional
          Atmospheric surface pressure, hPa. See also altitude.
      albedo : array-like, optional
          Ground surface albedo.
      pwater : array-like, optional
          Precipitable water, cm.
      ozone : array-like, optional
          Ozone vertical pathlength, atm-cm. Note: 1 atm-cm = 1000 DU
      beta : array-like, optional
          Angstrom's turbidity coefficient, i.e., AOD at 1000 nm. Input value
          will be checked for compliance with the mandatory interval [0, 2.2],
          and clipped if necessary.
      alpha : array-like, optional
          Angstrom's wavelength exponent, ideally obtained by linear
          regression of all available spectral AODs between 380 and 1020 nm.
          Input value will be checked for compliance with the mandatory
          interval [0, 2.5], and clipped if necessary.
      ssa : array-like, optional
          Aerosol single-scattering albedo at a representative wavelength of
          about 700 nm.  Will default to 0.92 if a negative value is input.
      asy : array-like, optional
          Aerosol asymmetry parameter. Since it tends to vary with wavelength
          and alpha, use a representative value for a wavelength of about
          700 nm and alpha about 1. Will default to 0.7 if a negative value
          is input.
      ecf : array-like, optional
          Sun-earth orbit eccentricity correction factor
      engine : numpy or numexpr

      Parameterization schemes
      ------------------------
      transmittance_scheme: string, optional
          broadband transmittances parameterization approach
             independent: the transmittances of the different atmospheric
                constituents are independent each other
             interdependent: the individual transmittances are interrelated.
                This approach is formally more correct but it is more complex
      csi_param: string, optional
          parameterization option for the circumsolar irradiance
             none: circumsolar irradiance is neglected
             sparta: native parameterization
      csi_hfov: float, optional
          half field of view angle (degrees) to evaluate CSI with
          csi_param=sparta

    Returns
    -------
      out : dictionary
          The (key, value) pairs in the dictionary are:
            dni : direct normal irradiance, in W/m2
            dhi : direct horizontal irradiance, in W/m2
            dif : diffuse horizontal irradiance, in W/m2
            ghi : global horizontal irradiance, in W/m2
            csi : circumsolar normal irradiance, in W/m2
    """

    if engine == 'numpy':
        logger.debug(f'running SPARTA with the {engine} engine')

        result = py_sparta(cosz=cosz, pressure=pressure, albedo=albedo, pwater=pwater,
                           ozone=ozone, beta=beta, alpha=alpha, ssa=ssa, asy=asy,
                           ecf=ecf, csi_param=csi_param, csi_hfov=csi_hfov,
                           transmittance_scheme=transmittance_scheme, as_dict=True)

    if engine == 'numexpr':
        logger.debug(f'running SPARTA with the {engine} engine')

        result = {}
        # result = ne_sparta(cosz=cosz, pressure=pressure, albedo=albedo, pwater=pwater,
        #                    ozone=ozone, beta=beta, alpha=alpha, ssa=ssa, asy=asy,
        #                    ecf=ecf, csi_param=csi_param, csi_hfov=csi_hfov,
        #                    transmittance_scheme=transmittance_scheme, as_dict=True)

    return result
