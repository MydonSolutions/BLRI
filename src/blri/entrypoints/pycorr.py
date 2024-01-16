import os, argparse, logging, time

import h5py
import numpy
from guppi import GuppiRawHandler

from blri import logger as blri_logger, parse, dsp
from blri.fileformats import telinfo as blri_telinfo, uvh5


def _get_jd(
    tbin,
    sampleperblk,
    piperblk,
    synctime,
    pktidx
):
    unix = synctime + (pktidx * (sampleperblk/piperblk)) * tbin
    return 2440587.5 + unix/86400


def main(arg_strs: list = None):
    parser = argparse.ArgumentParser(
        description="Correlate the data of a RAW file set, producing a UVH5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "raw_filepaths",
        type=str,
        nargs="+",
        help="The path to the GUPPI RAW file stem or of all files.",
    )
    parser.add_argument(
        "-t",
        "--telescope-info-filepath",
        type=str,
        required=True,
        help="The path to telescope information (YAML/TOML or BFR5['telinfo']).",
    )
    parser.add_argument(
        "-u",
        "--upchannelisation-rate",
        type=int,
        default=1,
        help="The upchannelisation rate.",
    )
    parser.add_argument(
        "-i",
        "--integration-rate",
        type=int,
        default=1,
        help="The integration rate.",
    )
    parser.add_argument(
        "-p",
        "--polarisations",
        type=str,
        default=None,
        help="The polarisation characters for each polarisation, as a string (e.g. 'xy').",
    )
    parser.add_argument(
        "-b", "--frequency-mhz-begin",
        type=float,
        default=None,
        help="The lowest frequency (MHz) of the fine-spectra to analyse (at least 1 channel will be processed).",
    )
    parser.add_argument(
        "-e", "--frequency-mhz-end",
        type=float,
        default=None,
        help="The highest frequency (MHz) of the fine-spectra to analyse (at least 1 channel will be processed).",
    )
    parser.add_argument(
        "-sp", "--frequency-selection-percentage",
        type=float,
        default=None,
        help="A decimal percentage of the bandwidth to process: [0.0, 1.0].",
    )
    parser.add_argument(
        "-sc", "--frequency-selection-center",
        type=float,
        default=0.5,
        help="Specifies the center of the sub-band, when processing a percentage of the bandwidth: [0.0, 1.0].",
    )
    parser.add_argument(
        "--output-filepath",
        type=str,
        default=None,
        help="The path to which the output will be written (instead of alongside the raw_filepath).",
    )
    parser.add_argument(
        "--cupy",
        action="store_true",
        help="Use cupy for DSP operations.",
    )
    parser.add_argument(
        "-T", "--dtype",
        type=str,
        default="float32",
        help="The numpy.dtype to load the GUPPI RAW data as (passed as an argument to `numpy.dtype()`)."
    )
    parser.add_argument(
        "--unsorted-raw-filepaths",
        action="store_true",
        help="Do not sort (lexicographically) the provided GUPPI RAW filepaths."
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

    if args.cupy:
        dsp.compute_with_cupy()
    else:
        dsp.compute_with_numpy()

    datablock_time_requirement = args.upchannelisation_rate

    telinfo = blri_telinfo.load_telescope_metadata(args.telescope_info_filepath)
    if len(args.raw_filepaths) == 1 and not os.path.exists(args.raw_filepaths[0]):
        # argument appears to be a singular stem, break it out of the list
        args.raw_filepaths = args.raw_filepaths[0]
    elif not args.unsorted_raw_filepaths:
        args.raw_filepaths.sort()
    guppi_handler = GuppiRawHandler(args.raw_filepaths)

    if args.output_filepath is None:
        input_dir, input_filename = os.path.split(guppi_handler._guppi_filepaths[0])
        output_filepath = os.path.join(input_dir, f"{os.path.splitext(input_filename)[0]}.uvh5")
    else:
        output_filepath = args.output_filepath

    blri_logger.debug(f"GUPPI RAW files: {guppi_handler._guppi_filepaths}")
    guppi_bytes_total = numpy.sum(list(map(os.path.getsize, guppi_handler._guppi_filepaths)))
    blri_logger.debug(f"Total GUPPI RAW bytes: {guppi_bytes_total/(10**6)} MB")

    guppi_blocks_iter = guppi_handler.blocks(astype=guppi_block_astype_dtype.type)
    guppi_header, guppi_data = next(guppi_blocks_iter)
    if hasattr(guppi_header, "antenna_names"):
        telinfo.antennas = blri_telinfo.filter_and_reorder_antenna_in_telinfo(
            telinfo,
            guppi_header.antenna_names
        ).antennas
    assert len(telinfo.antennas) == guppi_header.nof_antennas, f"len({telinfo.antennas}) != {guppi_header.nof_antennas}"

    ant_1_array, ant_2_array = uvh5.get_uvh5_ant_arrays(telinfo.antennas)
    num_bls = len(ant_1_array)

    nants, nchan, ntimes, npol = guppi_header.blockshape
    coarse_chan_bw = guppi_header.channel_bandwidth
    frequency_channel_0_mhz = guppi_header.observed_frequency - (nchan/2 + 0.5)*coarse_chan_bw

    upchan_bw = coarse_chan_bw/args.upchannelisation_rate
    frequencies_mhz = frequency_channel_0_mhz + numpy.arange(nchan*args.upchannelisation_rate)*upchan_bw

    frequency_lowest_mhz = min(frequencies_mhz[0], frequencies_mhz[-1])
    frequency_highest_mhz = max(frequencies_mhz[0], frequencies_mhz[-1])

    if args.frequency_selection_percentage is not None:
      frequencies_mhz_range = frequency_highest_mhz-frequency_lowest_mhz
      subband_mhz_center = args.frequency_selection_center*frequencies_mhz_range + frequency_lowest_mhz
      subband_mhz_range = args.frequency_selection_percentage*frequencies_mhz_range
      args.frequency_mhz_begin = subband_mhz_center - subband_mhz_range/2
      args.frequency_mhz_end = subband_mhz_center + subband_mhz_range/2

    if args.frequency_mhz_begin is None:
        args.frequency_mhz_begin = frequency_lowest_mhz
    elif args.frequency_mhz_begin < frequency_lowest_mhz:
        raise ValueError(f"Specified begin frequency is out of bounds: {frequency_lowest_mhz} Hz")

    if args.frequency_mhz_end is None:
        args.frequency_mhz_end = frequency_highest_mhz
    elif args.frequency_mhz_end > frequency_highest_mhz:
        raise ValueError(f"Specified end frequency is out of bounds: {frequency_highest_mhz} Hz")

    frequency_begin_fineidx = len(list(filter(lambda x: x<-upchan_bw, frequencies_mhz-args.frequency_mhz_begin)))
    frequency_end_fineidx = len(list(filter(lambda x: x<=0, frequencies_mhz-args.frequency_mhz_end)))
    assert frequency_begin_fineidx != frequency_end_fineidx, f"{frequency_begin_fineidx} == {frequency_end_fineidx}"

    blri_logger.info(f"Fine-frequency channel range: [{frequency_begin_fineidx}, {frequency_end_fineidx})")
    blri_logger.info(f"Fine-frequency range: [{frequencies_mhz[frequency_begin_fineidx]}, {frequencies_mhz[frequency_end_fineidx-1]}] MHz")

    frequency_begin_coarseidx = int(numpy.floor(frequency_begin_fineidx/args.upchannelisation_rate))
    frequency_end_coarseidx = int(numpy.ceil(frequency_end_fineidx/args.upchannelisation_rate))

    if frequency_end_coarseidx == frequency_begin_coarseidx:
        if frequency_end_coarseidx <= nchan:
            frequency_end_coarseidx += 1
        else:
            assert frequency_begin_coarseidx >= 1
            frequency_begin_coarseidx -= 1

    blri_logger.info(f"Coarse-frequency channel range: [{frequency_begin_coarseidx}, {frequency_end_coarseidx})")
    blri_logger.info(f"Coarse-frequency range: [{frequencies_mhz[frequency_begin_coarseidx*args.upchannelisation_rate]}, {frequencies_mhz[frequency_end_coarseidx*args.upchannelisation_rate - 1]}] MHz")
    assert frequency_begin_coarseidx != frequency_end_coarseidx

    frequencies_mhz = frequencies_mhz[frequency_begin_fineidx:frequency_end_fineidx]

    frequency_end_fineidx = frequency_end_fineidx - frequency_begin_coarseidx*args.upchannelisation_rate
    frequency_begin_fineidx = frequency_begin_fineidx - frequency_begin_coarseidx*args.upchannelisation_rate

    blri_logger.info(f"Fine-frequency relative channel range: [{frequency_begin_fineidx}, {frequency_end_fineidx})")
    blri_logger.info(f"Fine-frequency range: [{frequencies_mhz[0]}, {frequencies_mhz[-1]}] MHz")
    frequencies_mhz += 0.5 * upchan_bw

    polarisation_chars = guppi_header.get("POLS", args.polarisations)
    assert polarisation_chars is not None

    assert len(polarisation_chars) == npol
    polproducts = [
        f"{pol1}{pol2}"
        for pol1 in polarisation_chars for pol2 in polarisation_chars
    ]

    phase_center_radians = (
        parse.degrees_process(guppi_header.get("RA_PHAS", guppi_header.rightascension_string)) * numpy.pi / 12.0 ,
        parse.degrees_process(guppi_header.get("DEC_PHAS", guppi_header.declination_string)) * numpy.pi / 180.0 ,
    )

    timeperblk = guppi_data.shape[2]
    piperblk = guppi_header.nof_packet_indices_per_block
    dut1 = guppi_header.get("DUT1", 0.0)
    integration_time_per_spectrum_out = args.upchannelisation_rate*args.integration_rate*guppi_header.spectra_timespan
    blri_logger.debug(f"integration_time_per_spectrum_out: {integration_time_per_spectrum_out}")
    
    jd_time_running = _get_jd(
        guppi_header.spectra_timespan,
        ntimes,
        piperblk,
        guppi_header.time_unix_offset,
        guppi_header.packet_index
    )
    SEC_PER_DAY = 86400
    jd_time_running += 0.5*integration_time_per_spectrum_out/SEC_PER_DAY # jd_time_array holds average timestamp of spectrum, maintain offset
    
    jd_time_array = numpy.array((num_bls,), dtype='d')
    jd_time_array.fill(jd_time_running)
    integration_time = numpy.array((num_bls,), dtype='d')
    integration_time.fill(integration_time_per_spectrum_out)
    flags = numpy.zeros((num_bls, len(frequencies_mhz), len(polproducts)), dtype='?')
    nsamples = numpy.ones(flags.shape, dtype='d')

    ant_xyz = numpy.array([ant.position for ant in telinfo.antennas])
    antenna_numbers = [ant.number for ant in telinfo.antennas]
    baseline_ant_1_indices = [antenna_numbers.index(antnum) for antnum in ant_1_array]
    baseline_ant_2_indices = [antenna_numbers.index(antnum) for antnum in ant_2_array]
    lla = (telinfo.longitude_radians, telinfo.latitude_radians, telinfo.altitude)

    with h5py.File(output_filepath, "w") as f:

        uvh5_datasets = uvh5.uvh5_initialise(
            f,
            telinfo.telescope_name,
            guppi_header.telescope,  # instrument_name
            guppi_header.source_name,  # source_name,
            lla,
            telinfo.antennas,
            frequencies_mhz*1e6,
            polproducts,
            phase_center_radians,
            dut1=dut1
        )

        datablock_shape = list(guppi_data.shape)
        datablocks_per_requirement = datablock_time_requirement/datablock_shape[2]
        blri_logger.debug(f"Collects ceil({datablocks_per_requirement}) blocks for correlation.")
        datablock_shape[2] = numpy.ceil(datablocks_per_requirement)*datablock_shape[2]

        datablocks_queue = [guppi_data]
        data_spectra_count = guppi_data.shape[2]

        t = time.perf_counter_ns()
        t_start = t
        last_file_pos = 0
        datasize_processed = 0
        integration_count = 0
        # Integrate fine spectra in a separate buffer
        integration_buffer = dsp.compy.zeros(flags.shape, dtype="D")
        read_elapsed_s = 0.0
        concat_elapsed_s = 0.0
        t_progress = t
        while True:
            if data_spectra_count >= datablock_time_requirement:
                if len(datablocks_queue) == 1:
                    datablock = datablocks_queue[0]
                else: 
                    t = time.perf_counter_ns()
                    datablock = numpy.concatenate(
                        datablocks_queue,
                        axis=2  # concatenate in time
                    )
                    concat_elapsed_s += 1e-9*(time.perf_counter_ns() - t)
                datablocks_queue.clear()

                file_pos = guppi_handler._guppi_file_handle.tell()
                datasize_processed += file_pos - last_file_pos
                last_file_pos = file_pos
                progress = datasize_processed/guppi_bytes_total

                t = time.perf_counter_ns()
                progress_elapsed_s = 1e-9*(t - t_progress)
                t_progress = t

                total_elapsed_s = 1e-9*(t - t_start)
                blri_logger.info(f"Progress: {datasize_processed/10**6:0.3f}/{guppi_bytes_total/10**6:0.3f} MB ({100*progress:03.02f}%). Elapsed: {total_elapsed_s:0.3f} s, ETC: {total_elapsed_s*(1-progress)/progress:0.3f} s")
                blri_logger.debug(f"Read time: {read_elapsed_s} s ({100*read_elapsed_s/progress_elapsed_s:0.2f} %)")
                blri_logger.debug(f"Concat time: {concat_elapsed_s} s ({100*concat_elapsed_s/progress_elapsed_s:0.2f} %)")
                blri_logger.debug(f"Running throughput: {datasize_processed/(total_elapsed_s*10**6):0.3f} MB/s")

                read_elapsed_s = 0.0
                concat_elapsed_s = 0.0

            while datablock.shape[2] >= datablock_time_requirement:
                datablock_residual = datablock[:, :, datablock_time_requirement:, :]
                datablock = dsp.compy.array(
                    datablock[:, :, 0:datablock_time_requirement, :]
                )

                datablock_bytesize = datablock.size * datablock.itemsize

                t = time.perf_counter_ns()
                datablock = dsp.upchannelise(
                    datablock[:, frequency_begin_coarseidx:frequency_end_coarseidx, :, :],
                    args.upchannelisation_rate
                )[:, frequency_begin_fineidx:frequency_end_fineidx, :, :]

                elapsed_s = 1e-9*(time.perf_counter_ns() - t)
                blri_logger.debug(f"Channelisation: {datablock_bytesize/(elapsed_s*10**6)} MB/s")
                datablock_bytesize = datablock.size * datablock.itemsize

                t = time.perf_counter_ns()
                datablock = dsp.correlate(datablock)
                elapsed_s = 1e-9*(time.perf_counter_ns() - t)
                blri_logger.debug(f"Correlation: {datablock_bytesize/(elapsed_s*10**6)} MB/s")

                t = time.perf_counter_ns()
                assert datablock.shape[2] == 1
                integration_buffer += datablock.reshape(integration_buffer.shape)
                elapsed_s = 1e-9*(time.perf_counter_ns() - t)
                blri_logger.debug(f"Integration {integration_count}/{args.integration_rate}: {datablock_bytesize/(elapsed_s*10**6)} MB/s")
                integration_count += 1

                datablock = datablock_residual
                del datablock_residual

                if integration_count < args.integration_rate:
                    continue

                t = time.perf_counter_ns()
                uvh5.uvh5_write_chunk(
                    uvh5_datasets,
                    ant_1_array,
                    ant_2_array,
                    uvh5.get_uvw_array(
                        jd_time_array[0],
                        phase_center_radians,
                        ant_xyz,
                        lla,
                        baseline_ant_1_indices,
                        baseline_ant_2_indices,
                        dut1=dut1,
                    ),
                    jd_time_array,
                    integration_time,
                    integration_buffer.get() if dsp.cupy_enabled else integration_buffer,
                    flags,
                    nsamples,
                )
                elapsed_s = 1e-9*(time.perf_counter_ns() - t)
                blri_logger.debug(f"Write: {datablock_bytesize/(elapsed_s*10**6)} MB/s")
                blri_logger.debug(f"jd_time_running: {jd_time_running}")

                jd_time_running += integration_time_per_spectrum_out / SEC_PER_DAY
                jd_time_array.fill(jd_time_running)

                integration_buffer.fill(0.0)
                integration_count = 0

                datablocks_queue = [datablock]
                data_spectra_count = datablock.shape[2]

            _guppi_file_index = guppi_handler._guppi_file_index
            try:
                t = time.perf_counter_ns()
                guppi_header, guppi_data = next(guppi_blocks_iter)
                read_elapsed_s += 1e-9*(time.perf_counter_ns() - t)
                data_spectra_count += guppi_data.shape[2]
                datablocks_queue.append(guppi_data)
            except StopIteration:
                break
            if guppi_handler._guppi_file_index != _guppi_file_index:
                last_file_pos = 0

    return output_filepath
