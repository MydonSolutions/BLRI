import os
import yaml
from typing import List, Tuple, Optional
from typing_extensions import Annotated
from enum import Enum

import numpy
import h5py
import tomli as tomllib  # `tomllib` as of Python 3.11 (PEP 680)
from pydantic import BaseModel, BeforeValidator, field_serializer, model_validator


from blri import parse, coords
from blri import logger


class AntennaDetail(BaseModel):
    name: str
    number: int
    position: Tuple[float, float, float]
    diameter: Optional[float] = None


class AntennaPositionFrame(str, Enum):
    enu = "enu"
    xyz = "xyz"
    ecef = "ecef"


class TelescopeInformation(BaseModel):
    telescope_name: str
    longitude: Annotated[
        float,  # degrees
        BeforeValidator(parse.degrees_process),
    ]
    latitude: Annotated[
        float,  # degrees
        BeforeValidator(parse.degrees_process),
    ]
    altitude: float
    antenna_diameter: Optional[float] = None
    antenna_position_frame: AntennaPositionFrame
    antennas: List[AntennaDetail]

    longitude_radians = property(
        fget=lambda self: self.longitude * numpy.pi / 180,
        fset=None
    )
    latitude_radians = property(
        fget=lambda self: self.latitude * numpy.pi / 180,
        fset=None
    )

    @field_serializer('antenna_position_frame')
    def serialize_antenna_position_frame(self, apf: AntennaPositionFrame, _info):
        return apf.value

    @model_validator(mode='after')
    def ensure_antenna_position_frame_xyz(self) -> 'TelescopeInformation':
        for i, antenna in enumerate(self.antennas):
            if antenna.diameter is None:
                antenna.diameter = self.antenna_diameter
            assert antenna.diameter is not None, f"Antenna {antenna.name} does not have a diameter."

        if self.antenna_position_frame == AntennaPositionFrame.xyz:
            return self

        antenna_positions = numpy.array([
            antenna.position
            for antenna in self.antennas
        ])
        if self.antenna_position_frame == AntennaPositionFrame.ecef:
            coords.transform_antenna_positions_ecef_to_xyz(
                self.longitude_radians,
                self.latitude_radians,
                self.altitude,
                antenna_positions
            )
        if self.antenna_position_frame == AntennaPositionFrame.enu:
            coords.transform_antenna_positions_enu_to_xyz(
                self.longitude_radians,
                self.latitude_radians,
                self.altitude,
                antenna_positions
            )

        for i, antenna in enumerate(self.antennas):
            antenna.position = tuple(antenna_positions[i, :])

        self.antenna_position_frame = AntennaPositionFrame.xyz
        return self


def load_telescope_metadata(telescope_info_filepath) -> TelescopeInformation:
    """
    Parses TOML/YAML/BFR5 contents, returning a standardised TelescopeInformation.

    Ingested longitude and latitude values are expected to be in degrees.
    """
    _, telinfo_ext = os.path.splitext(telescope_info_filepath)
    if telinfo_ext in [".toml"]:
        with open(telescope_info_filepath, mode="rb") as f:
            telescope_info = tomllib.load(f)
    elif telinfo_ext in [".yaml", ".yml"]:
        with open(telescope_info_filepath, mode="r") as f:
            telescope_info = yaml.load(f, Loader=yaml.CSafeLoader)
    elif telinfo_ext in [".bfr5"]:
        with h5py.File(telescope_info_filepath, 'r') as f:
            telescope_info = {
                "telescope_name": f["telinfo"]["telescope_name"][()],
                "longitude": f["telinfo"]["longitude"][()],
                "latitude": f["telinfo"]["latitude"][()],
                "altitude": f["telinfo"]["altitude"][()],
                "antenna_position_frame": f["telinfo"]["antenna_position_frame"][()],
                "antennas": [
                    {
                        "number": antenna_number,
                        "name": f["telinfo"]["antenna_names"][i],
                        "position": f["telinfo"]["antenna_positions"][i],
                        "diameter": f["telinfo"]["antenna_diameters"][i],
                    }
                    for i, antenna_number in enumerate(
                        f["telinfo"]["antenna_numbers"]
                    )
                ]
            }
    else:
        raise ValueError(
            f"Unknown file format '{telinfo_ext}'"
            f" ({os.path.basename(telescope_info_filepath)})."
            f" Known formats are: yaml, toml, bfr5."
        )

    return TelescopeInformation(**telescope_info)


def filter_and_reorder_antenna_in_telinfo(
    telinfo: TelescopeInformation,
    antenna_names: List[str]
):
    antenna_telinfo = {
        antenna.name: antenna
        for antenna in telinfo.antennas
        if antenna.name in antenna_names
    }
    if len(antenna_telinfo) != len(antenna_names):
        raise ValueError(
            "Telescope information does not cover listed antenna: "
            f"{set(antenna_names).difference(set([ant.name for ant in telinfo.antennas]))}"
        )

    return telinfo.__class__(
        telescope_name=telinfo.telescope_name,
        longitude=telinfo.longitude,
        latitude=telinfo.latitude,
        altitude=telinfo.altitude,
        antenna_position_frame=telinfo.antenna_position_frame,
        antennas=[
            antenna_telinfo[antname]
            for antname in antenna_names
        ]
    )
