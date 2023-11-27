import unittest

import os, sys
from typing import List, Tuple, Optional

from pydantic import BaseModel

from blri.entrypoints.bfr5 import generate_for_raw as bfr5gen_main
from blri.fileformats.bfr5 import bfr5_differences
from blri.fileformats.telinfo import AntennaPositionFrame
from blri.tests.input_generation import GuppiRawSizeParameterSet, gen_guppi_input, gen_telinfo_input

class ScenarioParameterSet(BaseModel):
    position_frame: AntennaPositionFrame

    def to_id_str(self) -> str:
        return f"AntFrame: {self.position_frame}"

class Bfr5GenParameterSet(BaseModel):
    phase_center_radec_hrdeg: Optional[Tuple[float, float]]
    beams_radec_hrdeg: List[Tuple[float, float]]

    def to_id_str(self) -> str:
        return f"pc{self.phase_center_radec_hrdeg}b{'b'.join(map(str, self.beams_radec_hrdeg))}"

class EntrypointBfr5GenParameterSet(BaseModel):
    guppi_size_param: GuppiRawSizeParameterSet
    scenario_param_list: List[ScenarioParameterSet]
    bfr5gen_param_list: List[Bfr5GenParameterSet]
    
    @staticmethod
    def output_reference_id_str(
        guppi_size_param,
        bfr5gen_param
    ) -> str:
        return f"{guppi_size_param.to_id_str()}.{bfr5gen_param.to_id_str()}"
    
    
    def sub_parameter_sets(self, UnitTestInstance):
        for scenario_param in self.scenario_param_list:
            for bfr5gen_param in self.bfr5gen_param_list:
                output_reference_id = self.output_reference_id_str(self.guppi_size_param, bfr5gen_param)
                with UnitTestInstance.subTest(f"{output_reference_id} ({scenario_param.to_id_str()})"):
                    yield (
                        os.path.join(
                            os.path.dirname(__file__),
                            "references",
                            "bfr5gen",
                            f"test.{output_reference_id}.bfr5"
                        ),
                        scenario_param,
                        bfr5gen_param
                    )


TEST_CASES = [
    EntrypointBfr5GenParameterSet(
        guppi_size_param=GuppiRawSizeParameterSet(
            blockcount=20,
            blockshape=(3,16,128,2)
        ),
        scenario_param_list=[
            ScenarioParameterSet(
                position_frame=AntennaPositionFrame.xyz
            )
        ],
        bfr5gen_param_list=[
            Bfr5GenParameterSet(
                phase_center_radec_hrdeg=None,
                beams_radec_hrdeg=[]
            )
        ]
    ),
]

class Testbfr5genEntrypoint(unittest.TestCase):
    def run_entrypoint(
        self,
        filepath_telinfo: str,
        filepath_gr: str,
        bfr5gen_param: Bfr5GenParameterSet
    ):
        filepath_output = "test.bfr5"
        args = [
            "-t", filepath_telinfo,
            "--output-filepath", filepath_output,
            "-vvv"
        ] + [
            ("b " + ','.join(map(str, beam))).split(" ")
            for beam in bfr5gen_param.beams_radec_hrdeg
        ]
        if bfr5gen_param.phase_center_radec_hrdeg is not None:
            args += [
                "-p", bfr5gen_param.phase_center_radec_hrdeg
            ]
        args.append(filepath_gr)

        bfr5gen_main(map(str, args))
        return filepath_output
    
    @staticmethod
    def assert_bfr5_equality(
        filepath_a,
        filepath_b,
        atol=1e-8, rtol=1e-5
    ):
        group_diff_dict = bfr5_differences(
            filepath_a,
            filepath_b,
            atol=atol, rtol=rtol
        )
        
        for group, diffs in group_diff_dict.items():
            assert len(diffs) == 0, f"Group '{group}' mismatch on datasets: {diffs}"

    def output_and_reference_testcase_tuples(
        self,
        scenario_predicate=lambda _: True
    ):
        for test_params in TEST_CASES:
            guppi_input_path, grh = gen_guppi_input(test_params.guppi_size_param)
            for sub_paramset_tuple in test_params.sub_parameter_sets(self):
                filepath_reference_output, scenario_param, bfr5gen_param = sub_paramset_tuple
                if not scenario_predicate(scenario_param):
                    continue

                telinfo_input_path = gen_telinfo_input(grh, scenario_param.position_frame)
                filepath_output = self.run_entrypoint(
                    telinfo_input_path,
                    guppi_input_path,
                    bfr5gen_param
                )
                yield (
                    filepath_reference_output,
                    filepath_output
                )

    def test_entrypoint(self):
        for filepath_reference_output, filepath_output in self.output_and_reference_testcase_tuples():
            self.assert_bfr5_equality(
                filepath_reference_output,
                filepath_output
            )

    def write_reference_outputs(self):
        for filepath_reference_output, filepath_output in self.output_and_reference_testcase_tuples():
            os.replace(src=filepath_output, dst=filepath_reference_output)


if __name__ == '__main__':
    if sys.argv[-1] == "write_references":
        Testbfr5genEntrypoint().write_reference_outputs()
        exit(0)

    unittest.main(verbosity=3)
