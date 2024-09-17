import argparse
import logging

from blri import logger as blri_logger
from blri.fileformats.telinfo import AntennaPositionFrame
from blri.tests.input_generation import GuppiRawSizeParameterSet, gen_guppi_input, gen_telinfo_input

def input_gen():
    parser = argparse.ArgumentParser(
        description="Generate GUPPI RAW and associated telinfo files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-filepath",
        type=str,
        default="test",
        help="The path to output to.",
    )
    parser.add_argument(
        "-D", "--dimension-lengths",
        type=int,
        metavar=("A", "F", "T", "P"),
        nargs=4,
        help="The shape of each block as [A, F, T, P].",
    )
    parser.add_argument(
        "-N", "--number-of-blocks",
        type=int,
        help="The number of blocks in the GUPPI RAW file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the verbosity of the comparison (0=Silent, 1=Print Field Names, 2=Print Fields (Header only), 3=Print Fields)."
    )
    args = parser.parse_args()
    blri_logger.setLevel(
        [
            logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG
        ][args.verbose]
    )

    guppi_path, grh = gen_guppi_input(
        GuppiRawSizeParameterSet(
            blockcount=args.number_of_blocks,
            blockshape=(*args.dimension_lengths, )
        ),
        filepath=f"{args.output_filepath}.0000.raw"
    )
    blri_logger.debug(guppi_path)
    telinfo_path = gen_telinfo_input(
        grh,
        AntennaPositionFrame.xyz,
        filepath=f"{args.output_filepath}.telinfo.yaml"
    )
    blri_logger.info(telinfo_path)

if __name__ == "__main__":
    input_gen()