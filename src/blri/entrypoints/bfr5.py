from typing import Tuple, List
import os, sys, argparse, glob, time, json, logging
from blri.parse import degrees_process

import numpy
from guppi import GuppiRawHandler, header as guppi_header

from blri import logger as blri_logger
from blri.fileformats import telinfo as blri_telinfo, bfr5
from blri.times import julian_date_from_unix


def _base_arguments_parser():
    parser = argparse.ArgumentParser(
        description="A script that generates a BFR5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--telescope-info-filepath",
        type=str,
        required=True,
        help="The path to telescope information.",
    )
    parser.add_argument(
        "--output-filepath",
        type=str,
        default=None,
        help="The path to which the output will be written (instead of alongside the raw_filepath).",
    )
    parser.add_argument(
        "-m",
        "--mitigate-antenna",
        default=[],
        nargs="+",
        metavar=("antname"),
        help="The names of antenna to mitigate (associated calibration coefficients are set to zero)."
    )
    parser.add_argument(
        "-p",
        "--phase-center",
        default=None,
        metavar=("ra_hour,dec_deg"),
        help="The coordinates of the data's phase-center."
    )
    parser.add_argument(
        "raw_filepaths",
        type=str,
        nargs="+",
        help="The path to the GUPPI RAW file stem or of all files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the verbosity of the generation (0=Error, 1=Warn, 2=Info, 3=Debug)."
    )
    return parser

def _get_raw_metadata(raw_filepaths: List[str]):
    raw_header = None
    raw_block_count = 0
    for hdr in GuppiRawHandler(raw_filepaths).headers():
        if raw_block_count == 0:
            raw_header = hdr
        raw_block_count += 1
    
    
    # General RAW metadata
    nants, nchan, ntimes, npol = raw_header.blockshape
    schan = raw_header.observed_channels_offset

    start_time_unix = raw_header.time_unix_epoch_seconds
    block_time_span_s = raw_header.nof_packet_indices_per_block * raw_header.spectra_timespan * ntimes/raw_header.nof_packet_indices_per_block

    primary_center_radec_rad = (
        degrees_process(raw_header.get("RA_PRIM", raw_header.rightascension_hours)) * numpy.pi / 12.0,
        degrees_process(raw_header.get("DEC_PRIM", raw_header.declination_degrees)) * numpy.pi / 180.0
    )

    phase_center_radec_rad = (
        degrees_process(raw_header.get("RA_PHAS", raw_header.rightascension_hours)) * numpy.pi / 12.0,
        degrees_process(raw_header.get("DEC_PHAS", raw_header.declination_degrees)) * numpy.pi / 180.0
    )

    # find the observation channel0 frequency, which is the start of the frequency range
    frequency_channel_0_hz = raw_header.observed_frequency - (nchan/2 + schan)*raw_header.channel_bandwidth
    blri_logger.debug(f"frequency_channel_0_hz: {frequency_channel_0_hz}")
    frequencies_hz = (frequency_channel_0_hz + numpy.arange(schan+nchan)*raw_header.channel_bandwidth)*1e6
    blri_logger.debug(f"frequencies_hz: {frequencies_hz}")
    mid_chan = schan + (nchan//2)
    assert frequencies_hz[mid_chan] == raw_header.observed_frequency*1e6, f"frequencies_hz[{mid_chan}] = {frequencies_hz[mid_chan]} != {raw_header.observed_frequency*1e6} (OBSFREQ)"
    frequencies_hz += raw_header.channel_bandwidth*0.5e6

    times_unix = (start_time_unix + 0.5 * block_time_span_s) + numpy.arange(raw_block_count)*block_time_span_s

    return raw_header, frequencies_hz, times_unix, phase_center_radec_rad, primary_center_radec_rad


def _parse_base_arguments(args):
    blri_logger.setLevel(
        [
            logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG
        ][args.verbose]
    )
    if len(args.raw_filepaths) == 1 and not os.path.exists(args.raw_filepaths[0]):
        blri_logger.info(f"Given RAW filepath does not exist, assuming it is the stem.")
        args.raw_filepaths = glob.glob(f"{args.raw_filepaths[0]}*.raw")
        args.raw_filepaths.sort()
        blri_logger.debug(f"Found {args.raw_filepaths}.")

    raw_header, frequencies_hz, times_unix, phase_center_radec_rad, primary_center_radec_rad = _get_raw_metadata(args.raw_filepaths)
    telinfo = blri_telinfo.load_telescope_metadata(args.telescope_info_filepath)

    if hasattr(raw_header, "antenna_names"):
        telinfo.antennas = blri_telinfo.filter_and_reorder_antenna_in_telinfo(
            telinfo,
            raw_header.antenna_names
        ).antennas
    assert len(telinfo.antennas) == raw_header.nof_antennas, f"len({telinfo.antennas}) != {raw_header.nof_antennas}"

    input_dir, input_filename = os.path.split(args.raw_filepaths[0])
    if args.output_filepath is None:
        output_filepath = os.path.join(input_dir, f"{os.path.splitext(input_filename)[0]}.bfr5")
    else:
        output_filepath = args.output_filepath

    if args.phase_center is not None:
        (phase_center_ra, phase_center_dec) = args.phase_center.split(',')
        phase_center_radec_rad = (
            float(phase_center_ra) * numpy.pi / 12.0,
            float(phase_center_dec) * numpy.pi / 180.0,
        )
        
    nants, nchan, ntimes, npol = raw_header.blockshape
    calcoeff_bandpass = numpy.ones(
        (len(frequencies_hz), npol, nants)
    )*(1+0j)

    calcoeff_gain = numpy.ones(
        (npol, nants)
    )*(1+0j)
    antenna_names = [
        antenna.name
        for antenna in telinfo.antennas
    ]
    for ant_name in args.mitigate_antenna:
        if ant_name not in antenna_names:
            raise RuntimeError(f"Cannot mitigate antenna {ant_name} as it is not recorded in the file. Antenna names: {antenna_names}")
        index = antenna_names.index(ant_name)
        calcoeff_gain[:, index] = 0

    return raw_header, frequencies_hz, times_unix, phase_center_radec_rad, primary_center_radec_rad, telinfo, output_filepath, calcoeff_bandpass, calcoeff_gain

# def _add_arguments_targetselector(parser):
#     parser.add_argument(
#         "--redis-hostname",
#         type=str,
#         default="redishost",
#         help="The hostname of the Redis server.",
#     )
#     parser.add_argument(
#         "--redis-port",
#         type=int,
#         default=6379,
#         help="The port of the Redis server.",
#     )
#     parser.add_argument(
#         "--targets-redis-key-prefix",
#         type=str,
#         default="targets:VLA-COSMIC:vlass_array",
#         help="The prefix of the redis key holding targets in JSON."
#     )
#     parser.add_argument(
#         "--targets-redis-key-timestamp",
#         type=str,
#         default=None,
#         help="The identifying timestamp of the redis key holding targets in JSON. If not supplied, it is ascertained from the targets-redis-key-timestamp-rawkey value."
#     )
#     parser.add_argument(
#         "--targets-redis-key-timestamp-rawkey",
#         type=str,
#         default="PKTIDX",
#         choices=["PKTIDX", "PKTSTART"],
#         help="The RAW key to use to calculate the timestamp of the target-selector's output redis key. While PKTIDX is unique per block (and so is different per file), PKTSTART is uniform across blocks."
#     )
#     parser.add_argument(
#         "--take-targets",
#         type=int,
#         default=0,
#         help="The number of targets to form beams on from the redis key."
#     )
#     parser.add_argument(
#         "--take-targets-after",
#         type=int,
#         default=1,
#         help="The number of targets to skip. Typically 1 to skip the first target which is phase-center."
#     )
#     parser.add_argument(
#         "--target",
#         type=str,
#         default=[],
#         action="append",
#         help="A named target to retrieve using astroquery.jplhorizons."
#     )


def _add_arguments_beams(parser):
    parser.add_argument(
        "-b",
        "--beam",
        default=None,
        action="append",
        metavar=("ra_hour,dec_deg[,name]"),
        help="""The coordinates of a beam (optionally the name too). 
        Can be specified as sexagesimal.
        Begin both start/stop values with a sign ('+' or '-') for `--phase-center` relative values.
        """
    )


def _add_arguments_raster(parser):
    parser.add_argument(
        "--raster-ra",
        required=True,
        metavar=("ra_offset_start", "ra_offset_stop", "ra_offset_step"),
        nargs=3,
        type=str,
        help="""The right-ascension range (in hours) for a raster set of beams.
        Can be specified as sexagesimal.
        Begin both start/stop values with a sign ('+' or '-') for `--raster-center` relative values.
        Further prepend either start/stop value with an 's' to specify multiples of step size.
        Begin step value with '/' specify the step count instead of the step size.
        """
    )
    parser.add_argument(
        "--raster-dec",
        required=True,
        metavar=("dec_offset_start", "dec_offset_stop", "dec_offset_step"),
        nargs=3,
        type=str,
        help="""The declination range (in degrees) for a raster set of beams.
        Can be specified as sexagesimal (with ':').
        Begin both start/stop values with a sign ('+' or '-') for `--raster-center` relative values.
        End either start/stop integer value with an 's' to specify multiples of step size.
        Begin step value with '/' specify the step count instead of the step size.
        """
    )
    parser.add_argument(
        "--raster-center",
        default=["+0.0", "+0.0"],
        metavar=("ra_center", "dec_center"),
        nargs=2,
        type=str,
        help="""The center point of the raster (units hours, degrees).
        Can be specified as sexagesimal (with ':').
        Begin both start/stop values with a sign ('+' or '-') for phase-center relative values.
        Note that the default is the phase-center.
        """
    )
    parser.add_argument(
        "--include-stops",
        action="store_true",
        help="Include the stop value of the raster ranges."
    )


def _generate_bfr5_for_raw(
    raw_header,
    frequencies_hz: numpy.ndarray,
    times_unix: numpy.ndarray,
    phase_center: Tuple[float, float],
    primary_center: Tuple[float, float],
    telinfo: blri_telinfo.TelescopeInformation,
    output_filepath,
    calcoeff_bandpass: numpy.ndarray, # (nchan, npol, nants)
    calcoeff_gain: numpy.ndarray, # (npol, nants)
    beam_strs
):
    
    beams = {}
    for i, beam_str in enumerate(beam_strs):
        coords = beam_str.split(',')
        if len(coords) == 3:
            beam_name = coords[-1]
        else:
            beam_name = f"BEAM_{i}"
        if all(coord_str[0] in ["+", "-"] for coord_str in coords[0:2]):
            coords[0] = phase_center[0]*12.0/numpy.pi + degrees_process(coords[0])
            coords[1] = phase_center[1]*180.0/numpy.pi + degrees_process(coords[1])
        else:
            coords[0] = degrees_process(coords[0])
            coords[1] = degrees_process(coords[1])

        beams[beam_name] = (
            float(coords[0]) * numpy.pi / 12.0,
            float(coords[1]) * numpy.pi / 180.0,
        )

    nbeams = len(beams)
    if nbeams == 0:
        blri_logger.warning(f"No beam coordinates provided, forming a beam on phase-center.")
        beams["PHASE_CENTER"] = phase_center
        nbeams = 1
    
    beam_coordinates_string = "\n\t".join(f"{k}: ({v[0]*12.0/numpy.pi} h, {v[1]*180.0/numpy.pi} d)" for k, v in beams.items())
    blri_logger.info(f"Beam coordinates:\n\t{beam_coordinates_string}")

    antenna_telinfo = {
        antenna.name: antenna
        for antenna in telinfo.antennas
    }

    bfr5.write(
        output_filepath,
        raw_header.get("OBSID", "Unknown"),
        telinfo.telescope_name,
        "Unknown Instrument",
        beams,
        phase_center,
        (telinfo.longitude_radians, telinfo.latitude_radians, telinfo.altitude),
        antenna_telinfo,
        times_unix,
        frequencies_hz,
        calcoeff_bandpass,
        calcoeff_gain,
        dut1=raw_header.get("DUT1", 0.0),
        primary_center=primary_center,
        reference_antenna_name=raw_header.get("REFANT", None)
    )
    blri_logger.info(f"Output: {output_filepath}")
    return output_filepath


# def generate_targets_for_raw(arg_strs=None):
#     parser = _base_arguments_parser()
#     _add_arguments_beams(parser)
#     _add_arguments_targetselector(parser)
#     args = parser.parse_args(arg_strs)
    
#     raw_header, frequencies_hz, times_unix, phase_center, primary_center, telinfo, output_filepath, calcoeff_bandpass, calcoeff_gain = _parse_base_arguments(args)

#     beam_strs = []
#     if args.take_targets > 0:
#         redis_obj = redis.Redis(host=args.redis_hostname, port=args.redis_port)
#         if args.targets_redis_key_timestamp is None:
#             file_start_packet_index = raw_header["SYNCTIME"] + raw_header[args.targets_redis_key_timestamp_rawkey]
#             blri_logger.info(f"Targets key timestamp is taken from {args.targets_redis_key_timestamp_rawkey}: {file_start_packet_index}")
#             args.targets_redis_key_timestamp = file_start_packet_index

#         targets_redis_key = f"{args.targets_redis_key_prefix}:{args.targets_redis_key_timestamp}"
#         blri_logger.info(f"Accessing targets at {targets_redis_key}.")
#         targets = redis_obj.get(targets_redis_key)
#         if targets is None:
#             raise ValueError(f"No targets to retrieve at: {targets_redis_key}.")

#         targets = json.loads(targets)

#         for target in targets[args.take_targets_after : args.take_targets_after+args.take_targets]:
#             beam_strs.append(
#                 f"{target['ra']*24/360},{target['dec']},{target['source_id']}"
#             )

#     if args.beam is not None and len(args.beam) > 0:
#         beam_strs += list(b for b in args.beam)
    
#     jd_mean = julian_date_from_unix(numpy.mean(times_unix)) 
#     for target in args.target:
#         obj = Horizons(
#             id=target,
#             location={
#                 'lon': telinfo["longitude"]*180/numpy.pi,
#                 'lat': telinfo["latitude"]*180/numpy.pi,
#                 'elevation': telinfo["altitude"]/1000
#             },
#             epochs=jd_mean
#         )
#         retries = 5
#         while True:
#             try:
#                 retries -= 1
#                 eph = obj.ephemerides()
#                 break
#             except ValueError as err:
#                 if retries == 0:
#                     message = f"Could not get ephemerides for target: {target} @ {jd_mean} JD."
#                     blri_logger.error(message)
#                     raise RuntimeError(message) from err
#                 time.sleep(0.25)

#         for row in eph:
#             beam_strs.append(f"{row['RA']*12/180},{row['DEC']},{row['targetname']}")


#     if len(beam_strs) < args.take_targets:
#         blri_logger.warning(f"Could only take {len(beam_strs)} targets.")

#     return _generate_bfr5_for_raw(
#         raw_header,
#         frequencies_hz,
#         times_unix,
#         phase_center,
#         primary_center,
#         telinfo,
#         output_filepath,
#         calcoeff_bandpass,
#         calcoeff_gain,
#         beam_strs
#     )


def _parse_raster_coord_truple(truple, include_stops, relative_value):
    """
    Returns an iterable
    """
    
    start_units_not_steps = truple[0][0] != "s"
    if not start_units_not_steps:
        truple[0] = truple[0][1:]

    stop_units_not_steps = truple[1][0] != "s"
    if not stop_units_not_steps:
        truple[1] = truple[1][1:]

    truple_relative = all(map(lambda s: s[0] in ["+", "-"], truple[0:2]))
    step_sized_not_counted = truple[2][0] != "/"

    if not step_sized_not_counted and not all([start_units_not_steps, stop_units_not_steps]):
        raise ValueError(f"Cannot specify coordinate in units of steps when the step is specified as a count.")
    
    step = truple[2]
    if step_sized_not_counted:
        step = degrees_process(step)
    else:
        step = int(step[1:])

    if start_units_not_steps:
        start = degrees_process(truple[0])
    else:
        start = float(truple[0])*step
    if stop_units_not_steps:
        stop = degrees_process(truple[1])
    else:
        stop = float(truple[1])*step
    
    if truple_relative:
        start += relative_value
        stop += relative_value

    if step_sized_not_counted:
        if include_stops:
            stop += step
        return numpy.arange(start, stop, step)
    else:
        return numpy.linspace(start, stop, step)


def _parse_raster_coord(coord, primary_center):
    coord_relative = all(map(lambda s: s[0] in ["+", "-"], coord))
    coord = list(map(degrees_process, coord))
    if coord_relative:
        coord[0] += primary_center.ra.hourangle
        coord[1] += primary_center.dec.degree
    
    return (
        coord[0] * numpy.pi / 12.0 ,
        coord[1] * numpy.pi / 180.0 ,
    )


def generate_raster_for_raw(arg_strs: list = None):
    parser = _base_arguments_parser()
    _add_arguments_raster(parser)
    print(sys.argv[1:])
    args = parser.parse_args(arg_strs)

    raw_header, frequencies_hz, times_unix, phase_center, primary_center, telinfo, output_filepath, calcoeff_bandpass, calcoeff_gain = _parse_base_arguments(args)

    beam_strs = []
    raster_args = [args.raster_ra, args.raster_dec]
    assert all(raster_args), f"Must supply both raster arguments for raster beams to be generated"

    raster_relative_coord = _parse_raster_coord(args.raster_center, phase_center)
    
    args.raster_ra = _parse_raster_coord_truple(args.raster_ra, args.include_stops, raster_relative_coord.ra.hourangle)
    args.raster_dec = _parse_raster_coord_truple(args.raster_dec, args.include_stops, raster_relative_coord.dec.degree)

    for ra_index, ra in enumerate(args.raster_ra):
        for dec_index, dec in enumerate(args.raster_dec):
            beam_strs.append(f"{ra:0.15f},{dec:0.15f},raster_{ra_index}_{dec_index}")

    return _generate_bfr5_for_raw(
        raw_header,
        frequencies_hz,
        times_unix,
        phase_center,
        primary_center,
        telinfo,
        output_filepath,
        calcoeff_bandpass,
        calcoeff_gain,
        beam_strs
    )

def generate_for_raw(arg_strs: list = None):
    parser = _base_arguments_parser()
    _add_arguments_beams(parser)
    args = parser.parse_args(arg_strs)

    raw_header, frequencies_hz, times_unix, phase_center, primary_center, telinfo, output_filepath, calcoeff_bandpass, calcoeff_gain = _parse_base_arguments(args)

    beam_strs = []
    if args.beam is None:
        # scrape from RAW file RA_OFF%01d,DEC_OFF%01d
        key_enum = 0
        while True:
            ra_key = f"RA_OFF{key_enum}"
            dec_key = f"DEC_OFF{key_enum}"
            if not (ra_key in raw_header and dec_key in raw_header):
                break

            beam_strs.append(f"{raw_header[ra_key]},{raw_header[dec_key]},BEAM_{key_enum}")

            key_enum += 1
            if key_enum == 10:
                break

        blri_logger.info(f"Collected {key_enum} beam coordinates from the RAW header, in lieu of CLI provided beam coordinates.")
    elif len(args.beam) > 0:
        blri_logger.info(args.beam)
        beam_strs = list(b for b in args.beam)

    return _generate_bfr5_for_raw(
        raw_header,
        frequencies_hz,
        times_unix,
        phase_center,
        primary_center,
        telinfo,
        output_filepath,
        calcoeff_bandpass,
        calcoeff_gain,
        beam_strs
    )
