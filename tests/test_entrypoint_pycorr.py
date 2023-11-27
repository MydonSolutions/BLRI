import unittest

import os, sys
from typing import List

from pydantic import BaseModel

from blri.entrypoints.pycorr import main as pycorr_main
from blri.fileformats.uvh5 import uvh5_differences
from blri.fileformats.telinfo import AntennaPositionFrame
from blri.tests.input_generation import GuppiRawSizeParameterSet, gen_guppi_input, gen_telinfo_input

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
                            "pycorr",
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

    def output_and_reference_testcase_tuples(
        self,
        scenario_predicate=lambda _: True
    ):
        for test_params in TEST_CASES:
            guppi_input_path, grh = gen_guppi_input(test_params.guppi_size_param)
            for sub_paramset_tuple in test_params.sub_parameter_sets(self):
                filepath_reference_output, scenario_param, pycorr_param = sub_paramset_tuple
                if not scenario_predicate(scenario_param):
                    continue

                telinfo_input_path = gen_telinfo_input(grh, scenario_param.position_frame)
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
