import pyproj
import numpy


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


def transform_antenna_positions_enu_to_xyz(
    longitude_rad,
    latitude_rad,
    altitude,
    antenna_positions
):
    sin_long = numpy.sin(longitude_rad)
    cos_long = numpy.cos(longitude_rad)
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
