from typing import Tuple

import numpy
import yaml
from pydantic import BaseModel

from guppi.header import GuppiRawHeader
from guppi import GuppiRawHandler

from blri import coords
from blri.fileformats.telinfo import TelescopeInformation, AntennaPositionFrame, AntennaDetail


class GuppiRawSizeParameterSet(BaseModel):
    blockcount: int
    blockshape: Tuple[int, int, int, int]

    def to_id_str(self) -> str:
        return f"{self.blockcount}x{'_'.join(map(str, self.blockshape))}"

def gen_telinfo_input(
        grh,
        output_frame: AntennaPositionFrame,
        filepath = None,
        rng = None,
    ) -> str:
    filepath = filepath or "test_telinfo.yaml"
    rng = rng or numpy.random.default_rng(3141592653**2)

    telinfo = generate_telinfo(
        rng,
        range(grh.nof_antennas)
    )

    if output_frame != AntennaPositionFrame.xyz:
        antenna_positions = numpy.array([
            antenna.position
            for antenna in telinfo
        ])
        transform = {
            AntennaPositionFrame.ecef: coords.transform_antenna_positions_xyz_to_ecef,
            AntennaPositionFrame.enu: coords.transform_antenna_positions_xyz_to_enu
        }[output_frame]
        transform(
            telinfo.longitude,
            telinfo.latitude,
            telinfo.altitude,
            antenna_positions
        )
        for i, antenna in telinfo:
            antenna.position = antenna_positions[i, :]
        telinfo.antenna_position_frame = output_frame

    write_telinfo(
        filepath,
        telinfo.model_dump()
    )
    return filepath
    

def gen_guppi_input(
        guppi_size_param: GuppiRawSizeParameterSet,
        filepath = None,
        rng = None,
    ) -> str:
    filepath = filepath or "test.0000.raw"
    rng = rng or numpy.random.default_rng(3141592635**3)

    grh = generate_guppi_header(
        rng,
        guppi_size_param.blockshape
    )

    write_guppi_data(
        filepath,
        grh,
        rng,
        nof_blocks=guppi_size_param.blockcount
    )
    return filepath, grh


def generate_telinfo(rng, antenna_numbers):
    return TelescopeInformation(
        telescope_name="Seeded Random",
        longitude=-180 + rng.random()*180*2,
        latitude=-180/2 + rng.random()*180,
        altitude=800 + rng.random()*300,
        antenna_position_frame=AntennaPositionFrame.xyz,
        antennas=[
            AntennaDetail(
                name=f"ant{ant_enum:03d}",
                position=rng.random(3)*500,
                number=ant_enum,
                diameter=6,
            )
            for ant_enum in antenna_numbers
        ]
    )


def write_telinfo(filepath: str, telinfo: dict):
    with open(filepath, 'w') as fio:
        yaml.safe_dump(telinfo, fio)


def generate_guppi_header(
    rng,
    blockshape,
    **keyvalues
):
    grh = GuppiRawHeader(
        **keyvalues
    )

    grh.nof_antennas = blockshape[0]
    grh.observed_nof_channels = blockshape[0]*blockshape[1]
    grh.nof_polarizations = blockshape[3]
    grh.nof_bits = 8
    grh.blocksize = numpy.prod(blockshape)*2*grh.nof_bits//8

    grh.nof_packet_indices_per_block = grh.blockshape[2]

    grh.telescope = "Seeded Random"
    grh.source_name = "test_synth"
    grh.rightascension_hours = rng.random()*12
    grh.declination_degrees = rng.random()*180 - 90
    grh.observed_frequency = 1420.0  # MHz
    grh.channel_bandwidth = 0.5  # MHz
    grh["POLS"] = "xy"
    grh["DUT1"] = 0.0

    grh.spectra_timespan = 1/(grh.channel_bandwidth*1e6)
    grh.time_unix_offset = 1697963830
    grh.packet_index = 0

    return grh


def _get_guppi_integers(rng, shape, dtype):
    return rng.integers(
        numpy.iinfo(dtype).min,
        numpy.iinfo(dtype).max,
        size=shape,
        dtype=dtype
    )


def write_guppi_data(filepath, gr_header, rng, nof_blocks=20):
    cdtype = GuppiRawHandler.NUMPY_INTTYPE_COMPLEXVIEWTYPE_MAP[numpy.int8]
    gr_data = numpy.zeros(gr_header.blockshape, cdtype)

    for i in range(nof_blocks):
        gr_data[:]['re'] = _get_guppi_integers(
            rng,
            gr_header.blockshape,
            numpy.int8
        )
        gr_data[:]['im'] = _get_guppi_integers(
            rng,
            gr_header.blockshape,
            numpy.int8
        )

        GuppiRawHandler.write_to_file(
            filepath,
            gr_header,
            gr_data,
            file_open_mode='wb' if i == 0 else 'ab'
        )

        gr_header.packet_index += gr_header.nof_packet_indices_per_block