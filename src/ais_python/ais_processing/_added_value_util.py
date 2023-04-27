from math import radians, cos, sin, asin, sqrt, degrees, pi, atan2
from enum import Enum
from typing import Union, Tuple
import math


# mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
_AVG_EARTH_RADIUS_KM = 6371.0088


def haversine_distance(df):

    dist = haversine(
        (df.index[0], df.iloc[0]), (df.index[1], df.iloc[1]), unit=Unit.NAUTICAL_MILES
    )
    return dist


def calculate_bearing_single(df):
    pointA = df.index[0], df.iloc[0]
    pointB = df.index[1], df.iloc[1]
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (
        math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)
    )
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def spherical_distance(lat1, long1, lat2, long2):
    """'
    Adding the distance between two geographical locations
    the distance is calucualte on a sphere of radius r
    """
    phi1 = 0.5 * math.pi - lat1
    phi2 = 0.5 * math.pi - lat2
    # mean radius in meters
    r = 0.5 * (6378137 + 6356752)
    t = math.sin(phi1) * math.sin(phi2) * math.cos(long1 - long2) + math.cos(
        phi1
    ) * math.cos(phi2)
    return r * math.acos(t)


def ellipsoidal_distance(lat1, long1, lat2, long2):
    """'
    Adding the distance between two geographical locations
    the distance is calucualte on a ellipsoide with earth parms.

    this takes a lot of time. Be carefull when using.
    """

    # mean radius in meters
    a = 0.5 * (6378137 + 6356752)
    f = 1 / 298.257223563  # ellipsoid flattening
    b = (1 - f) * a
    tolerance = 1e-11  # to stop iteration

    phi1, phi2 = lat1, lat2
    U1 = math.atan((1 - f) * math.tan(phi1))
    U2 = math.atan((1 - f) * math.tan(phi2))
    L1, L2 = long1, long2
    L = L2 - L1

    lambda_old = L + 0

    while True:

        t = (math.cos(U2) * math.sin(lambda_old)) ** 2
        t += (
            math.cos(U1) * math.sin(U2)
            - math.sin(U1) * math.cos(U2) * math.cos(lambda_old)
        ) ** 2
        sin_sigma = t**0.5
        cos_sigma = math.sin(U1) * math.sin(U2) + math.cos(U1) * math.cos(
            U2
        ) * math.cos(lambda_old)
        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha = (
            math.cos(U1)
            * math.cos(U2)
            * math.sin(lambda_old)
            / (sin_sigma + 0.00000001)
        )
        cos_sq_alpha = 1 - sin_alpha**2
        cos_2sigma_m = cos_sigma - 2 * math.sin(U1) * (math.sin(U2) + 0.00000001) / (
            cos_sq_alpha + 0.00000001
        )
        C = f * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha)) / 16

        t = sigma + C * sin_sigma * (
            cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m**2)
        )
        lambda_new = L + (1 - C) * f * sin_alpha * t
        if abs(lambda_new - lambda_old) <= tolerance:
            break
        else:
            lambda_old = lambda_new

    u2 = cos_sq_alpha * ((a**2 - b**2) / b**2)
    A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    t = cos_2sigma_m + 0.25 * B * (cos_sigma * (-1 + 2 * cos_2sigma_m**2))
    t -= (
        (B / 6)
        * cos_2sigma_m
        * (-3 + 4 * sin_sigma**2)
        * (-3 + 4 * cos_2sigma_m**2)
    )
    delta_sigma = B * sin_sigma * t
    return b * A * (sigma - delta_sigma)


class Unit(Enum):
    """
    Enumeration of supported units.
    The full list can be checked by iterating over the class; e.g.
    the expression `tuple(Unit)`.
    """

    KILOMETERS = "km"
    METERS = "m"
    MILES = "mi"
    NAUTICAL_MILES = "nmi"
    FEET = "ft"
    INCHES = "in"
    RADIANS = "rad"
    DEGREES = "deg"


class Direction(Enum):
    """
    Enumeration of supported directions.
    The full list can be checked by iterating over the class; e.g.
    the expression `tuple(Direction)`.
    Angles expressed in radians.
    """

    NORTH = 0
    NORTHEAST = pi * 0.25
    EAST = pi * 0.5
    SOUTHEAST = pi * 0.75
    SOUTH = pi
    SOUTHWEST = pi * 1.25
    WEST = pi * 1.5
    NORTHWEST = pi * 1.75


# Unit values taken from http://www.unitconversion.org/unit_converter/length.html
_CONVERSIONS = {
    Unit.KILOMETERS: 1.0,
    Unit.METERS: 1000.0,
    Unit.MILES: 0.621371192,
    Unit.NAUTICAL_MILES: 0.539956803,
    Unit.FEET: 3280.839895013,
    Unit.INCHES: 39370.078740158,
    Unit.RADIANS: 1 / _AVG_EARTH_RADIUS_KM,
    Unit.DEGREES: (1 / _AVG_EARTH_RADIUS_KM) * (180.0 / pi),
}


def get_avg_earth_radius(unit):
    unit = Unit(unit)
    return _AVG_EARTH_RADIUS_KM * _CONVERSIONS[unit]


def _normalize(lat: float, lon: float) -> Tuple[float, float]:
    """
    Normalize point to [-90, 90] latitude and [-180, 180] longitude.
    """
    lat = (lat + 90) % 360 - 90
    if lat > 90:
        lat = 180 - lat
        lon += 180
    lon = (lon + 180) % 360 - 180
    return lat, lon


def _ensure_lat_lon(lat: float, lon: float):
    """
    Ensure that the given latitude and longitude have proper values. An exception is raised if they are not.
    """
    if lat < -90 or lat > 90:
        raise ValueError(f"Latitude {lat} is out of range [-90, 90]")
    if lon < -180 or lon > 180:
        raise ValueError(f"Longitude {lon} is out of range [-180, 180]")


def haversine(point1, point2, unit=Unit.KILOMETERS, normalize=False):
    """Calculate the great-circle distance between two points on the Earth surface.
    Takes two 2-tuples, containing the latitude and longitude of each point in decimal degrees,
    and, optionally, a unit of length.
    :param point1: first point; tuple of (latitude, longitude) in decimal degrees
    :param point2: second point; tuple of (latitude, longitude) in decimal degrees
    :param unit: a member of haversine.Unit, or, equivalently, a string containing the
                 initials of its corresponding unit of measurement (i.e. miles = mi)
                 default 'km' (kilometers).
    :param normalize: if True, normalize the points to [-90, 90] latitude and [-180, 180] longitude.
    Example: ``haversine((45.7597, 4.8422), (48.8567, 2.3508), unit=Unit.METERS)``
    Precondition: ``unit`` is a supported unit (supported units are listed in the `Unit` enum)
    :return: the distance between the two points in the requested unit, as a float.
    The default returned unit is kilometers. The default unit can be changed by
    setting the unit parameter to a member of ``haversine.Unit``
    (e.g. ``haversine.Unit.INCHES``), or, equivalently, to a string containing the
    corresponding abbreviation (e.g. 'in'). All available units can be found in the ``Unit`` enum.
    """

    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # normalize points or ensure they are proper lat/lon, i.e., in [-90, 90] and [-180, 180]
    if normalize:
        lat1, lng1 = _normalize(lat1, lng1)
        lat2, lng2 = _normalize(lat2, lng2)
    else:
        _ensure_lat_lon(lat1, lng1)
        _ensure_lat_lon(lat2, lng2)

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = radians(lat1)
    lng1 = radians(lng1)
    lat2 = radians(lat2)
    lng2 = radians(lng2)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2

    return 2 * get_avg_earth_radius(unit) * asin(sqrt(d))


def haversine_vector(array1, array2, unit=Unit.KILOMETERS, comb=False, normalize=False):
    """
    The exact same function as "haversine", except that this
    version replaces math functions with numpy functions.
    This may make it slightly slower for computing the haversine
    distance between two points, but is much faster for computing
    the distance between two vectors of points due to vectorization.
    """
    try:
        import numpy
    except ModuleNotFoundError:
        return "Error, unable to import Numpy,\
        consider using haversine instead of haversine_vector."

    # ensure arrays are numpy ndarrays
    if not isinstance(array1, numpy.ndarray):
        array1 = numpy.array(array1)
    if not isinstance(array2, numpy.ndarray):
        array2 = numpy.array(array2)

    # ensure will be able to iterate over rows by adding dimension if needed
    if array1.ndim == 1:
        array1 = numpy.expand_dims(array1, 0)
    if array2.ndim == 1:
        array2 = numpy.expand_dims(array2, 0)

    # Asserts that both arrays have same dimensions if not in combination mode
    if not comb:
        if array1.shape != array2.shape:
            raise IndexError(
                "When not in combination mode, arrays must be of same size. If mode is required, use comb=True as argument."
            )

    # normalize points or ensure they are proper lat/lon, i.e., in [-90, 90] and [-180, 180]
    if normalize:
        array1 = numpy.array([_normalize(p[0], p[1]) for p in array1])
        array2 = numpy.array([_normalize(p[0], p[1]) for p in array2])
    else:
        [_ensure_lat_lon(p[0], p[1]) for p in array1]
        [_ensure_lat_lon(p[0], p[1]) for p in array2]

    # unpack latitude/longitude
    lat1, lng1 = array1[:, 0], array1[:, 1]
    lat2, lng2 = array2[:, 0], array2[:, 1]

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = numpy.radians(lat1)
    lng1 = numpy.radians(lng1)
    lat2 = numpy.radians(lat2)
    lng2 = numpy.radians(lng2)

    # If in combination mode, turn coordinates of array1 into column vectors for broadcasting
    if comb:
        lat1 = numpy.expand_dims(lat1, axis=0)
        lng1 = numpy.expand_dims(lng1, axis=0)
        lat2 = numpy.expand_dims(lat2, axis=1)
        lng2 = numpy.expand_dims(lng2, axis=1)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = (
        numpy.sin(lat * 0.5) ** 2
        + numpy.cos(lat1) * numpy.cos(lat2) * numpy.sin(lng * 0.5) ** 2
    )

    return 2 * get_avg_earth_radius(unit) * numpy.arcsin(numpy.sqrt(d))


def inverse_haversine(
    point, distance, direction: Union[Direction, float], unit=Unit.KILOMETERS
):

    lat, lng = point
    lat, lng = map(radians, (lat, lng))
    d = distance
    r = get_avg_earth_radius(unit)
    brng = direction.value if isinstance(direction, Direction) else direction

    return_lat = asin(sin(lat) * cos(d / r) + cos(lat) * sin(d / r) * cos(brng))
    return_lng = lng + atan2(
        sin(brng) * sin(d / r) * cos(lat), cos(d / r) - sin(lat) * sin(return_lat)
    )

    return_lat, return_lng = map(degrees, (return_lat, return_lng))
    return return_lat, return_lng
