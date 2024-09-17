# latitude: 34:04:43.0",
# longitude "-107:37:04.0",
# altitude 2124,

# latitude: 34:04:43.729443
# longitude: -107:37:6.003085
# altitude: 2114.8787108091637,

import argparse, logging, sys

import blri
from blri import coords
from blri.coords import pyproj
import numpy

def ecef2xyz(inputs, settings):
    assert len(inputs) == 3
    
    transformer = pyproj.Proj.from_proj(
        pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
        pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
    )
    print(transformer.transform(
        *inputs
    ))

def xyz2ecef(inputs, settings):
    assert len(inputs) == 3
    
    transformer = pyproj.Proj.from_proj(
        pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
        pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
    )
    print(transformer.transform(
        *inputs
    ))

def coord_conversion_cli(arg_values=None):
    conversions = {
        "ecef2xyz": ecef2xyz,
        "xyz2ecef": xyz2ecef
    }

    parser = argparse.ArgumentParser(
        description="A script that performs coordinate transformations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "conversion",
        choices=conversions.keys(),
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=True,
        help="The input values."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the verbosity of the generation (0=Error, 1=Warn, 2=Info, 3=Debug)."
    )
    
    args = parser.parse_args(arg_values if arg_values is not None else sys.argv[1:])
    blri.logger.setLevel(
        [
            logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG
        ][args.verbose]
    )

    conversions[args.conversion](args.inputs, None)

if __name__ == "__main__":
    coord_conversion_cli()