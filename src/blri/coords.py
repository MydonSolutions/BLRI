from typing import Tuple

import pyproj
import numpy
import erfa

eraASTROM = Tuple[Tuple[float, Tuple[float, float, float], Tuple[float, float, float], float, Tuple[float, float, float], float, Tuple[Tuple[ float, float, float], Tuple[ float, float, float], Tuple[float, float, float]], float, float, float], float, float, float, float]

def transform_antenna_positions_ecef_to_xyz(
    longitude_rad,
    latitude_rad,
    altitude,
    antenna_positions
):
    transformer = pyproj.Proj.from_proj(
        pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
        pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
    )
    telescopeCenterXyz = transformer.transform(
        longitude_rad*180.0/numpy.pi,  # expects degrees
        latitude_rad*180.0/numpy.pi,  # expects degrees
        altitude,
    )
    for i in range(antenna_positions.shape[0]):
        antenna_positions[i, :] -= telescopeCenterXyz


def transform_antenna_positions_xyz_to_ecef(
    longitude_rad,
    latitude_rad,
    altitude,
    antenna_positions
):
    transformer = pyproj.Proj.from_proj(
        pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
        pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
    )
    telescopeCenterXyz = transformer.transform(
        longitude_rad*180.0/numpy.pi,  # expects degrees
        latitude_rad*180.0/numpy.pi,  # expects degrees
        altitude,
    )
    for i in range(antenna_positions.shape[0]):
        antenna_positions[i, :] += telescopeCenterXyz


def transform_antenna_positions_enu_to_xyz(
    longitude_rad,
    latitude_rad,
    altitude,
    antenna_positions
):
    sin_lat = numpy.sin(latitude_rad)
    cos_lat = numpy.cos(latitude_rad)

    for ant in range(antenna_positions.shape[0]):
        # RotX(latitude) anti-clockwise
        x_ = antenna_positions[ant, 0]
        y = cos_lat*antenna_positions[ant, 1] - (-sin_lat)*antenna_positions[ant, 2]
        z = (-sin_lat)*antenna_positions[ant, 1] + cos_lat*antenna_positions[ant, 2]

        # RotY(latitude) clockwise
        x = cos_lat*x_ + sin_lat*z
        z = -sin_lat*x_ + cos_lat*z

        # Permute (YZX) to (XYZ)
        antenna_positions[ant, 0] = z
        antenna_positions[ant, 1] = x
        antenna_positions[ant, 2] = y


def transform_antenna_positions_xyz_to_enu(longitude_rad, latitude_rad, altitude, antenna_positions):
    sin_long = numpy.sin(longitude_rad)
    cos_long = numpy.cos(longitude_rad)
    sin_lat = numpy.sin(latitude_rad)
    cos_lat = numpy.cos(latitude_rad)

    enus = numpy.zeros(antenna_positions.shape, dtype=numpy.float64)
    for ant in range(antenna_positions.shape[0]):
        # RotZ(longitude) anti-clockwise
        x = cos_long*antenna_positions[ant, 0] - (-sin_long)*antenna_positions[ant, 1]
        y = (-sin_long)*antenna_positions[ant, 0] + cos_long*antenna_positions[ant, 1]
        z = antenna_positions[ant, 2]

        # RotY(latitude) clockwise
        x_ = x
        x = cos_lat*x_ + sin_lat*z
        z = -sin_lat*x_ + cos_lat*z

        # Permute (UEN) to (ENU)
        enus[ant, 0] = y
        enus[ant, 1] = z
        enus[ant, 2] = x

    return enus


def compute_uvw_from_enu(
        time_jd: float,
        source_radec_rad: Tuple[float, float],
        ant_enu_coordinates: numpy.ndarray,
        longlatalt_rad: Tuple[float, float, float],
        dut1: float = 0.0,
        astrom: eraASTROM = None
    ):
    """Computes UVW antenna coordinates with respect to reference

    Args:
        time_jd: julian date of the coordinates
        source_radec_rad: tuple {ra, dec} radians
        ant_enu_coordinates: numpy.ndarray
            Antenna ENU coordinates. This is indexed as (antenna_number, enu)
        longlatalt_rad: tuple Reference Coordinates (radians)
            Longitude, Latitude, Altitude. The antenna_coordinates must have
            this component in them.
        dut1: Delta UTC-UT1
        astrom: eraASTROM
            erfa.apco13 generated astrom value to reuse.

    Returns:
        The UVW coordinates in metres of each antenna. This
        is indexed as (antenna_number, uvw)
    """

    if astrom is not None:
        ri, di = erfa.atciq(
            source_radec_rad[0], source_radec_rad[1],
            0, 0, 0, 0,
            astrom
        )
        aob, zob, ha_rad, dec_rad, rob = erfa.atioq(
            ri, di,
            astrom
        )
    else:
        aob, zob, ha_rad, dec_rad, rob, eo = erfa.atco13(
            source_radec_rad[0], source_radec_rad[1],
            0, 0, 0, 0,
            time_jd, 0,
            dut1,
            *longlatalt_rad,
            0, 0,
            0, 0, 0, 0
        )
        
    sin_hangle = numpy.sin(ha_rad)
    cos_hangle = numpy.cos(ha_rad)
    sin_declination = numpy.sin(dec_rad)
    cos_declination = numpy.cos(dec_rad)
    sin_latitude = numpy.sin(longlatalt_rad[1])
    cos_latitude = numpy.cos(longlatalt_rad[1])

    uvws = numpy.zeros(ant_enu_coordinates.shape, dtype=numpy.float64)

    for ant in range(ant_enu_coordinates.shape[0]):
        # RotX(latitude) anti-clockwise
        x = ant_enu_coordinates[ant, 0]
        y = cos_latitude*ant_enu_coordinates[ant, 1] - (-sin_latitude)*ant_enu_coordinates[ant, 2]
        z = (-sin_latitude)*ant_enu_coordinates[ant, 1] + cos_latitude*ant_enu_coordinates[ant, 2]

        # RotY(hour_angle) clockwise
        x_ = x
        x = cos_hangle*x_ + sin_hangle*z
        z = -sin_hangle*x_ + cos_hangle*z

        # RotX(declination) clockwise
        uvws[ant, 0] = x
        uvws[ant, 1] = cos_declination*y - sin_declination*z
        uvws[ant, 2] = sin_declination*y + cos_declination*z

    return uvws

def compute_uvw_from_xyz(
        time_jd: float,
        source_radec_rad: Tuple[float, float],
        ant_coordinates: numpy.ndarray,
        longlatalt_rad: Tuple[float, float, float],
        dut1: float = 0.0,
        astrom: eraASTROM = None
    ):
    """Computes UVW antenna coordinates with respect to reference

    Args:
        time_jd: julian date of the coordinates
        source_radec_rad: tuple {ra, dec} radians
        ant_coordinates: numpy.ndarray
            Antenna XYZ coordinates, relative to reference position. This is indexed as (antenna_number, xyz)
        longlatalt_rad: tuple Reference Coordinates (radians)
            Longitude, Latitude, Altitude. The antenna_coordinates must have
            this component in them.
        dut1: Delta UTC-UT1
        astrom: eraASTROM
            erfa.apco13 generated astrom value to reuse.

    Returns:
        The UVW coordinates in metres of each antenna. This
        is indexed as (antenna_number, uvw)
    """

    if astrom is not None:
        ri, di = erfa.atciq(
            source_radec_rad[0], source_radec_rad[1],
            0, 0, 0, 0,
            astrom
        )
        aob, zob, ha_rad, dec_rad, rob = erfa.atioq(
            ri, di,
            astrom
        )
    else:
        aob, zob, ha_rad, dec_rad, rob, eo = erfa.atco13(
            source_radec_rad[0], source_radec_rad[1],
            0, 0, 0, 0,
            time_jd, 0,
            dut1,
            *longlatalt_rad,
            0, 0,
            0, 0, 0, 0
        )
        
    sin_long_minus_hangle = numpy.sin(longlatalt_rad[0]-ha_rad)
    cos_long_minus_hangle = numpy.cos(longlatalt_rad[0]-ha_rad)
    sin_declination = numpy.sin(dec_rad)
    cos_declination = numpy.cos(dec_rad)

    uvws = numpy.zeros(ant_coordinates.shape, dtype=numpy.float64)

    for ant in range(ant_coordinates.shape[0]):
        # RotZ(long-ha) anti-clockwise
        x = cos_long_minus_hangle*ant_coordinates[ant, 0] - (-sin_long_minus_hangle)*ant_coordinates[ant, 1]
        y = (-sin_long_minus_hangle)*ant_coordinates[ant, 0] + cos_long_minus_hangle*ant_coordinates[ant, 1]
        z = ant_coordinates[ant, 2]

        # RotY(declination) clockwise
        x_ = x
        x = cos_declination*x_ + sin_declination*z
        z = -sin_declination*x_ + cos_declination*z

        # Permute (WUV) to (UVW)
        uvws[ant, 0] = y
        uvws[ant, 1] = z
        uvws[ant, 2] = x

    return uvws
