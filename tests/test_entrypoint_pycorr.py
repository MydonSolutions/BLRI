import unittest

import os, sys
import yaml
from typing import List, Tuple

import numpy
from pydantic import BaseModel
from guppi.header import GuppiRawHeader
from guppi import GuppiRawHandler

from blri import coords
from blri.entrypoints.pycorr import main as pycorr_main
from blri.fileformats.uvh5 import uvh5_differences
from blri.fileformats.telinfo import TelescopeInformation, AntennaPositionFrame, AntennaDetail


class GuppiRawSizeParameterSet(BaseModel):
    blockcount: int
    blockshape: Tuple[int, int, int, int]

    def to_id_str(self) -> str:
        return f"{self.blockcount}x{'_'.join(map(str, self.blockshape))}"

class ScenarioParameterSet(BaseModel):
    position_frame: AntennaPositionFrame
    cupy_enabled: bool

    def to_id_str(self) -> str:
        return f"CuPy: {self.cupy_enabled}, AntFrame: {self.position_frame}"

class PycorrParameterSet(BaseModel):
    upchannelisation_rate: int
    integration_rate: int
    numpy_dtype: str

    def to_id_str(self) -> str:
        return f"u{self.upchannelisation_rate}i{self.integration_rate}"

class EntrypointPycorrParameterSet(BaseModel):
    guppi_size_param: GuppiRawSizeParameterSet
    scenario_param_list: List[ScenarioParameterSet]
    pycorr_param_list: List[PycorrParameterSet]
    
    @staticmethod
    def output_reference_id_str(
        guppi_size_param,
        pycorr_param
    ) -> str:
        return f"{guppi_size_param.to_id_str()}.{pycorr_param.to_id_str()}"
    
    
    def sub_parameter_sets(self, UnitTestInstance):
        for scenario_param in self.scenario_param_list:
            for pycorr_param in self.pycorr_param_list:
                output_reference_id = self.output_reference_id_str(self.guppi_size_param, pycorr_param)
                with UnitTestInstance.subTest(f"{output_reference_id} ({scenario_param.to_id_str()})"):
                    yield (
                        os.path.join(
                            os.path.dirname(__file__),
                            "references",
                            f"test.{output_reference_id}.uvh5"
                        ),
                        scenario_param,
                        pycorr_param
                    )


TEST_CASES = [
    EntrypointPycorrParameterSet(
        guppi_size_param=GuppiRawSizeParameterSet(
            blockcount=20,
            blockshape=(3,16,128,2)
        ),
        scenario_param_list=[
            ScenarioParameterSet(
                position_frame=AntennaPositionFrame.xyz,
                cupy_enabled=cupy
            )
            for cupy in [False, True]
        ],
        pycorr_param_list=[
            PycorrParameterSet(
                upchannelisation_rate=4,
                integration_rate=128,
                numpy_dtype="float"       
            )
        ]
    ),
    EntrypointPycorrParameterSet(
        guppi_size_param=GuppiRawSizeParameterSet(
            blockcount=20,
            blockshape=(8,16,16,2)
        ),
        scenario_param_list=[
            ScenarioParameterSet(
                position_frame=AntennaPositionFrame.xyz,
                cupy_enabled=cupy
            )
            for cupy in [False, True]
        ],
        pycorr_param_list=[
            PycorrParameterSet(
                upchannelisation_rate=2,
                integration_rate=128,
                numpy_dtype="double"       
            )
        ]
    ),
    # 7 MB output reference file
    # EntrypointPycorrParameterSet(
    #     guppi_size_param=GuppiRawSizeParameterSet(
    #         blockcount=16,
    #         blockshape=(2,2,128,2)
    #     ),
    #     scenario_param_list=[
    #         ScenarioParameterSet(
    #             position_frame=AntennaPositionFrame.xyz,
    #             cupy_enabled=cupy
    #         )
    #         for cupy in [False, True]
    #     ],
    #     pycorr_param_list=[
    #         PycorrParameterSet(
    #             upchannelisation_rate=512,
    #             integration_rate=4,
    #             numpy_dtype="double"       
    #         )
    #     ]
    # )
]

class TestPycorrEntrypoint(unittest.TestCase):

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
            gr_data[:]['re'] = TestPycorrEntrypoint._get_guppi_integers(
                rng,
                gr_header.blockshape,
                numpy.int8
            )
            gr_data[:]['im'] = TestPycorrEntrypoint._get_guppi_integers(
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

    def run_entrypoint(
        self,
        filepath_telinfo: str,
        filepath_gr: str,
        pycorr_param: PycorrParameterSet,
        cupy: bool,
    ):
        filepath_output = "test.uvh5"
        args = [
            "-t", filepath_telinfo,
            "-u", pycorr_param.upchannelisation_rate,
            "-i", pycorr_param.integration_rate,
            "-T", pycorr_param.numpy_dtype,
            "--output-filepath", filepath_output,
            filepath_gr
        ]
        if cupy:
            args.insert(0, "--cupy")

        pycorr_main(map(str, args))
        return filepath_output
    
    @staticmethod
    def assert_uvh5_equality(
        filepath_a,
        filepath_b,
        atol=1e-8, rtol=1e-5
    ):
        header_fields_diff, data_fields_diff = uvh5_differences(
            filepath_a,
            filepath_b,
            atol=atol, rtol=rtol
        )
        
        header_fields_diff = set(header_fields_diff).difference(
            set(["history"])
        )
        assert len(header_fields_diff) == 0, f"Header mismatch on datasets: {list(header_fields_diff)}"
        assert len(data_fields_diff) == 0, f"Data mismatch on datasets: {data_fields_diff}"

    @staticmethod
    def gen_telinfo_input(grh, output_frame: AntennaPositionFrame) -> str:
        filepath_telinfo = "test_telinfo.yaml"
        rng = numpy.random.default_rng(3141592653**2)

        telinfo = TestPycorrEntrypoint.generate_telinfo(
            rng,
            range(grh.nof_antennas)
        )

        if output_frame != AntennaPositionFrame.xyz:
            antenna_positions = numpy.array([
                antenna.position
                for antenna in TestPycorrEntrypoint.telinfo
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

        TestPycorrEntrypoint.write_telinfo(
            filepath_telinfo,
            telinfo.model_dump()
        )
        return filepath_telinfo
        
    @staticmethod
    def gen_guppi_input(guppi_size_param: GuppiRawSizeParameterSet) -> str:
        filepath_gr = "test.0000.raw"
        rng = numpy.random.default_rng(3141592635**3)

        grh = TestPycorrEntrypoint.generate_guppi_header(
            rng,
            guppi_size_param.blockshape
        )

        TestPycorrEntrypoint.write_guppi_data(
            filepath_gr,
            grh,
            rng,
            nof_blocks=guppi_size_param.blockcount
        )
        return filepath_gr, grh

    def output_and_reference_testcase_tuples(
        self,
        scenario_predicate=lambda scen_param: True
    ):
        for test_params in TEST_CASES:
            guppi_input_path, grh = TestPycorrEntrypoint.gen_guppi_input(test_params.guppi_size_param)
            for sub_paramset_tuple in test_params.sub_parameter_sets(self):
                filepath_reference_output, scenario_param, pycorr_param = sub_paramset_tuple
                if not scenario_predicate(scenario_param):
                    continue

                telinfo_input_path = TestPycorrEntrypoint.gen_telinfo_input(grh, scenario_param.position_frame)
                filepath_output = self.run_entrypoint(
                    telinfo_input_path,
                    guppi_input_path,
                    pycorr_param,
                    scenario_param.cupy_enabled
                )
                yield (
                    filepath_reference_output,
                    filepath_output
                )

    def test_entrypoint(self):
        for filepath_reference_output, filepath_output in self.output_and_reference_testcase_tuples():
            self.assert_uvh5_equality(
                filepath_reference_output,
                filepath_output
            )

    def write_reference_outputs(self):
        for filepath_reference_output, filepath_output in self.output_and_reference_testcase_tuples(
            scenario_predicate=lambda scenario_param: scenario_param.cupy_enabled == False
        ):
            os.replace(src=filepath_output, dst=filepath_reference_output)


if __name__ == '__main__':
    if sys.argv[-1] == "write_references":
        TestPycorrEntrypoint().write_reference_outputs()
        exit(0)

    unittest.main(verbosity=3)
