import os, re, argparse, logging

import h5py
import numpy
from guppi import GuppiRawHandler

from blri import logger as blri_logger, dsp, interferometry
from blri.fileformats import telinfo as blri_telinfo

def upchannelise_frequencies(frequencies, rate):
    fine_frequencies = numpy.zeros((len(frequencies), rate), dtype=numpy.float64)
    chan_bw = 0
    for coarse_chan_i in range(len(frequencies)-1):
        chan_bw = frequencies[coarse_chan_i+1] - frequencies[coarse_chan_i]
        fine_frequencies[coarse_chan_i, :] = numpy.linspace(
            frequencies[coarse_chan_i],
            frequencies[coarse_chan_i+1],
            rate,
            endpoint=False
        )
    fine_frequencies[-1, :] = numpy.linspace(
        frequencies[-1],
        frequencies[-1]+chan_bw,
        rate,
        endpoint=False
    )
    return fine_frequencies.flatten()

def guppiheader_get_unix_midblock(guppiheader):
    ntime = guppiheader.nof_spectra_per_block
    synctime = guppiheader.get("SYNCTIME", 0)
    pktidx = guppiheader.get("PKTIDX", 0)
    tbin = guppiheader.get("TBIN", 1.0/guppiheader.get("CHAN_BW", 0.0))
    piperblk = guppiheader.get("PIPERBLK", ntime)
    return synctime + (pktidx + piperblk) * ((tbin * ntime)/piperblk)

def index_of_time(times, t):
    for i, ti in enumerate(times):
        if ti == t:
            return i
        if ti > t:
            assert i != 0, f"Time {t} appears to be before the start of times: {times[0]}"
            return i

    assert False, f"Time {t} appears to be past the end of times: {times[-1]}"


def main(arg_strs: list = None):
    parser = argparse.ArgumentParser(
        description="Generate a (RAW, BFR5):(Filterbank) input:output pair of beamforming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "bfr5_filepath",
        type=str,
        help="The path to the BFR5 file.",
    )
    parser.add_argument(
        "raw_filepaths",
        type=str,
        nargs="+",
        help="The path to the GUPPI RAW file stem or of all files.",
    )
    parser.add_argument(
        "-u",
        "--upchannelisation-rate",
        type=int,
        default=1,
        help="The upchannelisation rate.",
    )
    parser.add_argument(
        "-P",
        "--postbeamform-upchannelise",
        action="store_true",
        help="Upchannelise after beamformation, as opposed to pre-beamformation upchannelisation."
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        default=None
    )
    parser.add_argument(
        "-T", "--dtype",
        type=str,
        default="float32",
        help="The numpy.dtype to load the GUPPI RAW data as (passed as an argument to `numpy.dtype()`)."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the verbosity of the generation (0=Error, 1=Warn, 2=Info (progress statements), 3=Debug)."
    )
    args = parser.parse_args(arg_strs)
    blri_logger.setLevel(
        [
            logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG
        ][args.verbose]
    )
    guppi_block_astype_dtype = numpy.dtype(args.dtype)

    telinfo = blri_telinfo.load_telescope_metadata(args.bfr5_filepath)
    if len(args.raw_filepaths) == 1 and not os.path.exists(args.raw_filepaths[0]):
        # argument appears to be a singular stem, break it out of the list
        args.raw_filepaths = args.raw_filepaths[0]
    elif not args.unsorted_raw_filepaths:
        args.raw_filepaths.sort()
    guppi_handler = GuppiRawHandler(args.raw_filepaths)

    bfr5 = h5py.File(args.bfr5_filepath, "r")

    guppi_stem = re.match(r"(.*)\.\d{4}.raw", os.path.basename(guppi_handler._guppi_filepaths[0])).group(1)
    if args.output_directory is None:
        args.output_directory = os.path.dirname(guppi_handler._guppi_filepaths[0])
    
    bfr5Dims = (
        bfr5["diminfo"]["nants"][()],
        bfr5["diminfo"]["nchan"][()],
        bfr5["diminfo"]["ntimes"][()],
        bfr5["diminfo"]["npol"][()],
    )
    beam_radec_rad_list = [
        (
            bfr5["beaminfo"]["ras"][beamIdx],
            bfr5["beaminfo"]["decs"][beamIdx],
        )
        for beamIdx in range(bfr5["diminfo"]["nbeams"][()])
    ]
    boresight_radec_rad = (
        bfr5["obsinfo"]["phase_center_ra"][()],
        bfr5["obsinfo"]["phase_center_dec"][()]
    )

    blri_logger.debug("Beam Coordinates (RA, DEC):")
    for i, beam_radec in enumerate(beam_radec_rad_list):
        blri_logger.debug(f"\t{i}: {beam_radec}")

    frequencies = bfr5["obsinfo"]["freq_array"][guppi_header["SCHAN"]:] * 1e9
    times = bfr5["delayinfo"]["time_array"][:]
    times_jd = bfr5["delayinfo"]["jds"][:]

    if args.postbeamform_upchannelise:
        phasor_frequencies = frequencies
    else:
        phasor_frequencies = upchannelise_frequencies(
            frequencies,
            args.upchannelisation_rate
        )
    
    phasor_frequencies += (phasor_frequencies[2] - phasor_frequencies[1])/2

    blri_logger.debug(f"Generating Phasors...")
    blri_logger.debug(phasor_frequencies*1e-9)
    delays_ns = interferometry.delays(
        numpy.array([ant.position for ant in telinfo.antennas]),
        boresight_radec_rad,
        beam_radec_rad_list,
        times_jd,
        (telinfo.longitude_radians, telinfo.latitude_radians, telinfo.altitude),
        reference_antenna_index = 0,
    )

    bfr5_delays = bfr5["delayinfo"]["delays"][:]
    recipe_delays_agreeable = numpy.isclose(bfr5_delays, delays_ns, atol=0.0001)
    # recipe_delays_agreeable = bfr5_delays == delays
    if not recipe_delays_agreeable.all():
        blri_logger.warning(f"The delays in the provided recipe file do not match the calculated delays:\n{recipe_delays_agreeable}")
        # exit(1)
        blri_logger.warning(f"Using calculated delays:\n{delays_ns}")
        blri_logger.warning(f"Not given delays:\n{bfr5_delays}")
    else:
        blri_logger.info("The recipe file's delays match the calculated delays.")

    phasor_coeffs = interferometry.phasors_from_delays(
        delays_ns,
        phasor_frequencies,
        bfr5["calinfo"]["cal_all"][guppi_header["SCHAN"]:guppi_header["SCHAN"]+guppi_header["OBSNCHAN"]//guppi_header["NANTS"]], # [Antenna, Frequency-channel, Polarization]
    )


    block_index = 1
    block_times_index = 0
    file_open_mode = "wb"
    for guppi_header, guppi_data in guppi_handler.blocks(astype=guppi_block_astype_dtype.type):

        blockdims = guppi_data.shape
        for dim in [0,2,3]:
            assert bfr5Dims[dim] == blockdims[dim], f"#{block_index}: {bfr5Dims} != {blockdims}"

        if args.postbeamform_upchannelise:
            beam_inputdata = guppi_data
        else:
            beam_inputdata = dsp.upchannelise(
                guppi_data,
                args.upchannelisation_rate
            )

        try:
            block_times_index = index_of_time(
                times,
                guppiheader_get_unix_midblock(guppi_header)
            )
        except:
            pass
        beams = dsp.beamform(
            beam_inputdata,
            phasor_coeffs[:, :, :, block_times_index:block_times_index+1, :]
        )

        if args.postbeamform_upchannelise:
            beams = dsp.upchannelise(
                beams,
                args.upchannelisation_rate
            )
        
        guppi_header["DATATYPE"] = "FLOAT"
        guppi_header["TBIN"] *= args.upchannelisation_rate
        guppi_header["CHAN_BW"] /= args.upchannelisation_rate
        GuppiRawHandler.write_to_file(
            os.path.join(args.output_directory, f"{guppi_stem}-beamformed.raw"),
            guppi_header,
            beams.astype(numpy.complex64),
            file_open_mode=file_open_mode
        )
        file_open_mode = "ab"
        block_index += 1
