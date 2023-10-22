import unittest

import os
import filecmp
import yaml

import numpy

from blri.entrypoints.pycorr import main as pycorr_main
from blri.fileformats.telinfo import TelescopeInformation, AntennaPositionFrame, AntennaDetail
from guppi.header import GuppiRawHeader
from guppi import GuppiRawHandler


class TestEntrypointPycorr(unittest.TestCase):

    @staticmethod
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

    @staticmethod
    def write_telinfo(filepath: str, telinfo: dict):
        with open(filepath, 'w') as fio:
            yaml.safe_dump(telinfo, fio)

    @staticmethod
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
        grh.declination_degrees = rng.random()*360 - 180
        grh.declination_degrees = rng.random()*360 - 180
        grh.observed_frequency = 1420.0  # MHz
        grh.channel_bandwidth = 0.5  # MHz
        grh["POLS"] = "xy"
        grh["DUT1"] = 0.0

        grh.spectra_timespan = 1/(grh.channel_bandwidth*1e6)
        grh.time_unix_offset = 1697963830
        grh.packet_index = 0

        return grh

    @staticmethod
    def _get_guppi_integers(rng, shape, dtype):
        return rng.integers(
            numpy.iinfo(dtype).min,
            numpy.iinfo(dtype).max,
            size=shape,
            dtype=dtype
        )

    @staticmethod
    def write_guppi_data(filepath, gr_header, rng, nof_blocks=20):
        cdtype = GuppiRawHandler.NUMPY_INTTYPE_COMPLEXVIEWTYPE_MAP[numpy.int8]
        gr_data = numpy.zeros(gr_header.blockshape, cdtype)

        for i in range(nof_blocks):
            gr_data[:]['re'] = TestEntrypointPycorr._get_guppi_integers(
                rng,
                gr_header.blockshape,
                numpy.int8
            )
            gr_data[:]['im'] = TestEntrypointPycorr._get_guppi_integers(
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

    def test_entrypoint(
        self,
        blockcount=20,
        blockshape=(3, 16, 128, 2),
        upchannelisation_rate=1024,
        integration_rate=64,
        cupy=False
    ):
        rng = numpy.random.default_rng(3141592635**3)
        filepath_reference_output = os.path.join(
            os.path.dirname(__file__),
            "references",
            f"test.{blockcount}x{'_'.join(map(str,blockshape))}.u{upchannelisation_rate}i{integration_rate}.uvh5"
        )
        filepath_telinfo = "test_telinfo.yaml"
        filepath_gr = "test.0000.raw"
        filepath_output = "test.uvh5"

        grh = self.generate_guppi_header(
            rng,
            blockshape
        )

        self.write_telinfo(
            filepath_telinfo,
            self.generate_telinfo(
                rng,
                range(grh.nof_antennas)
            ).model_dump()
        )

        self.write_guppi_data(
            filepath_gr,
            grh,
            rng,
            nof_blocks=blockcount
        )

        args = [
            "-t", filepath_telinfo,
            "-u", upchannelisation_rate,
            "-i", integration_rate,
            "--output-filepath", filepath_output,
            filepath_gr
        ]
        if cupy:
            args.insert(0, "--cupy")

        pycorr_main(map(str, args))

        assert filecmp.cmp(
            filepath_output,
            filepath_reference_output,
            shallow=False
        )


if __name__ == '__main__':
    unittest.main()
