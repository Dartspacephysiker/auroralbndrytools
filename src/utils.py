import pandas as pd
import numpy as np

def subsol(datetimes):
    """ 
    calculate subsolar point at given datetime(s)

    returns:
      subsol_lat  -- latitude(s) of the subsolar point
      subsol_lon  -- longiutde(s) of the subsolar point

    The code is vectorized, so it should be fast.

    After Fortran code by: 961026 A. D. Richmond, NCAR

    Documentation from original code:
    Find subsolar geographic latitude and longitude from date and time.
    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994).
    Usable for years 1601-2100, inclusive.  According to the Almanac, 
    results are good to at least 0.01 degree latitude and 0.025 degree 
    longitude between years 1950 and 2050.  Accuracy for other years 
    has not been tested.  Every day is assumed to have exactly
    86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored:  their effect is below the accuracy threshold of
    the algorithm.

    Added by SMH 2020/04/03 (from Kalle's code stores!)
    """

    # use pandas DatetimeIndex for fast access to year, month day etc...
    if hasattr(datetimes, '__iter__'): 
        datetimes = pd.DatetimeIndex(datetimes)
    else:
        datetimes = pd.DatetimeIndex([datetimes])

    year = np.array(datetimes.year)
    # day of year:
    doy  = date_to_doy(datetimes.month, datetimes.day, dates.is_leapyear(year))
    # seconds since start of day:
    ut   = datetimes.hour * 60.**2 + datetimes.minute*60. + datetimes.second 
 
    yr = year - 2000

    if year.max() >= 2100 or year.min() <= 1600:
        raise ValueError('subsol.py: subsol invalid after 2100 and before 1600')

    nleap = np.floor((year-1601)/4.)
    nleap = nleap - 99

    # exception for years <= 1900:
    ncent = np.floor((year-1601)/100.)
    ncent = 3 - ncent
    nleap[year <= 1900] = nleap[year <= 1900] + ncent[year <= 1900]

    l0 = -79.549 + (-.238699*(yr-4*nleap) + 3.08514e-2*nleap)

    g0 = -2.472 + (-.2558905*(yr-4*nleap) - 3.79617e-2*nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut/86400. - 1.5) + doy

    # Addition to Mean longitude of Sun since January 1 of IYR:
    lf = .9856474*df

    # Addition to Mean anomaly since January 1 of IYR:
    gf = .9856003*df

    # Mean longitude of Sun:
    l = l0 + lf

    # Mean anomaly:
    g = g0 + gf
    grad = g*np.pi/180.

    # Ecliptic longitude:
    lmbda = l + 1.915*np.sin(grad) + .020*np.sin(2.*grad)
    lmrad = lmbda*np.pi/180.
    sinlm = np.sin(lmrad)

    # Days (including fraction) since 12 UT on January 1 of 2000:
    n = df + 365.*yr + nleap

    # Obliquity of ecliptic:
    epsilon = 23.439 - 4.e-7*n
    epsrad  = epsilon*np.pi/180.

    # Right ascension:
    alpha = np.arctan2(np.cos(epsrad)*sinlm, np.cos(lmrad)) * 180./np.pi

    # Declination:
    delta = np.arcsin(np.sin(epsrad)*sinlm) * 180./np.pi

    # Subsolar latitude:
    sbsllat = delta

    # Equation of time (degrees):
    etdeg = l - alpha
    nrot = np.round(etdeg/360.)
    etdeg = etdeg - 360.*nrot

    # Apparent time (degrees):
    aptime = ut/240. + etdeg    # Earth rotates one degree every 240 s.

    # Subsolar longitude:
    sbsllon = 180. - aptime
    nrot = np.round(sbsllon/360.)
    sbsllon = sbsllon - 360.*nrot

    return sbsllat, sbsllon


def date_to_doy(month, day, leapyear = False):
    """ return day of year (DOY) at given month, day

        month and day -- can be arrays, but must have equal shape
        leapyear      -- can be array of equal shape or scalar

        return value  --  doy, with same shape as month and day
                          but always an array: shape (1,) if input is scalar

        The code is vectorized, so it should be relatively fast. 

        KML 2016-04-20
    """

    month = np.array(month, ndmin = 1)
    day   = np.array(day, ndmin = 1)

    if type(leapyear) == bool:
        leapyear = np.full_like(day, leapyear, dtype = bool)

    # check that shapes match
    if month.shape != day.shape:
        raise ValueError('date2ody: month and day must have the same shape')

    # check that month in [1, 12]
    if month.min() < 1 or month.max() > 12:
        raise ValueError('month not in [1, 12]')

    # check if day < 1
    if day.min() < 1:
        raise ValueError('date2doy: day must not be less than 1')

    # flatten arrays:
    shape = month.shape
    month = month.flatten()
    day   = day.flatten()

    # check if day exceeds days in months
    days_in_month    = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    days_in_month_ly = np.array([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    if ( (np.any(day[~leapyear] > days_in_month   [month[~leapyear]])) | 
         (np.any(day[ leapyear] > days_in_month_ly[month[ leapyear]])) ):
        raise ValueError('date2doy: day must not exceed number of days in month')

    cumdaysmonth = np.cumsum(days_in_month[:-1])

    # day of year minus possibly leap day:
    doy = cumdaysmonth[month - 1] + day
    # add leap day where appropriate:
    doy[month >= 3] = doy[month >= 3] + leapyear[month >= 3]

    return doy.reshape(shape)
