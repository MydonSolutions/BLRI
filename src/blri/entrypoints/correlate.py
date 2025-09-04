import os, argparse, logging, time
from typing import Optional, Union

import h5py
import numpy

from blri import logger as blri_logger, dsp
from blri.fileformats import telinfo as blri_telinfo, uvh5
from blri.fileformats.guppiraw import InputGuppiIterator
from blri.fileformats.stamps import InputStampIterator
from blri.times import julian_date_from_unix


class CorrelationIterator:
    def __init__(
        self,
        inputhandler: Union[InputGuppiIterator, InputStampIterator],
        frequency_selection_center: Optional[float] = None, # decimal percentage
        frequency_selection_percentage: Optional[float] = None, # decimal percentage
        frequency_mhz_begin: Optional[float] = None,
        frequency_mhz_end: Optional[float] = None,
        upchannelisation_rate: int = 1,
        integration_rate: int = 1,
        invert_correlation_conjugation: Optional[bool] = False,
        invert_uvw_baselines: Optional[bool] = False,
        polarisations: Optional[str] = None,
    ):
        self._inputhandler = inputhandler
        self.upchannelisation_rate = upchannelisation_rate
        self.integration_rate = integration_rate
        self.invert_correlation_conjugation = invert_correlation_conjugation
        self.invert_uvw_baselines = invert_uvw_baselines
        
        self.metadata = inputhandler.metadata(polarisation_chars=polarisations)

        observed_frequency_bottom_mhz = self.metadata.observed_frequency_mhz - (self.metadata.nof_channel/2)*self.metadata.channel_bandwidth_mhz

        upchan_bw = self.metadata.channel_bandwidth_mhz/upchannelisation_rate
        # offset to center of channels is deferred
        frequencies_mhz = observed_frequency_bottom_mhz + numpy.arange(self.metadata.nof_channel*upchannelisation_rate)*upchan_bw

        frequency_lowest_mhz = min(frequencies_mhz[0], frequencies_mhz[-1])
        frequency_highest_mhz = max(frequencies_mhz[0], frequencies_mhz[-1])

        if frequency_selection_percentage is not None:
            frequencies_mhz_range = frequency_highest_mhz-frequency_lowest_mhz
            subband_mhz_center = frequency_selection_center*frequencies_mhz_range + frequency_lowest_mhz
            subband_mhz_range = frequency_selection_percentage*frequencies_mhz_range
            frequency_mhz_begin = subband_mhz_center - subband_mhz_range/2
            frequency_mhz_end = subband_mhz_center + subband_mhz_range/2

        if frequency_mhz_begin is None:
            frequency_mhz_begin = frequency_lowest_mhz
        elif frequency_mhz_begin < frequency_lowest_mhz:
            raise ValueError(f"Specified begin frequency is out of bounds: {frequency_lowest_mhz} Hz")

        if frequency_mhz_end is None:
            frequency_mhz_end = frequency_highest_mhz
        elif frequency_mhz_end > frequency_highest_mhz:
            raise ValueError(f"Specified end frequency is out of bounds: {frequency_highest_mhz} Hz")

        frequency_begin_fineidx = len(list(filter(lambda x: x<-upchan_bw, frequencies_mhz-frequency_mhz_begin)))
        frequency_end_fineidx = len(list(filter(lambda x: x<=0, frequencies_mhz-frequency_mhz_end)))
        assert frequency_begin_fineidx != frequency_end_fineidx, f"{frequency_begin_fineidx} == {frequency_end_fineidx}"

        blri_logger.info(f"Fine-frequency channel range: [{frequency_begin_fineidx}, {frequency_end_fineidx})")
        blri_logger.info(f"Fine-frequency range: [{frequencies_mhz[frequency_begin_fineidx]}, {frequencies_mhz[frequency_end_fineidx-1]}] MHz")

        self.frequency_begin_coarseidx = int(numpy.floor(frequency_begin_fineidx/upchannelisation_rate))
        self.frequency_end_coarseidx = int(numpy.ceil(frequency_end_fineidx/upchannelisation_rate))

        if self.frequency_end_coarseidx == self.frequency_begin_coarseidx:
            if self.frequency_end_coarseidx <= self.metadata.nof_channel:
                self.frequency_end_coarseidx += 1
            else:
                assert self.frequency_begin_coarseidx >= 1
                self.frequency_begin_coarseidx -= 1

        blri_logger.info(f"Coarse-frequency channel range: [{self.frequency_begin_coarseidx}, {self.frequency_end_coarseidx})")
        blri_logger.info(f"Coarse-frequency range: [{frequencies_mhz[self.frequency_begin_coarseidx*upchannelisation_rate]}, {frequencies_mhz[self.frequency_end_coarseidx*upchannelisation_rate - 1]}] MHz")
        assert self.frequency_begin_coarseidx != self.frequency_end_coarseidx

        self.frequencies_mhz = frequencies_mhz[frequency_begin_fineidx:frequency_end_fineidx]

        self.frequency_end_fineidx = frequency_end_fineidx - self.frequency_begin_coarseidx*upchannelisation_rate
        self.frequency_begin_fineidx = frequency_begin_fineidx - self.frequency_begin_coarseidx*upchannelisation_rate

        blri_logger.info(f"Fine-frequency relative channel range: [{frequency_begin_fineidx}, {frequency_end_fineidx})")
        blri_logger.info(f"Fine-frequency range: [{frequencies_mhz[0]}, {frequencies_mhz[-1]}] MHz")
        # offset to center of channels
        self.frequencies_mhz += 0.5 * upchan_bw

    
    def data(self):        
        inputdata_iter = self._inputhandler.data()
        inputdata = next(inputdata_iter) # premature check to ensure data can be accessed

        datablock_time_requirement = self.upchannelisation_rate

        datablock_shape = list(inputdata.shape)
        datablocks_per_requirement = datablock_time_requirement/datablock_shape[2]
        blri_logger.debug(f"Collects ceil({datablocks_per_requirement}) blocks for correlation.")
        datablock_shape[2] = numpy.ceil(datablocks_per_requirement)*datablock_shape[2]

        data_antenna_count = inputdata.shape[0]
        data_spectra_count = inputdata.shape[2]
        # while gathering blocks of data, `datablock` is shape (Block, A, F, T, P)
        datablock = inputdata.reshape([1, *inputdata.shape])

        t = time.perf_counter_ns()
        t_start = t
        last_data_pos = 0
        datasize_processed = 0
        integration_count = 0
        # Integrate fine spectra in a separate buffer
        correlation_shape = (data_antenna_count*(data_antenna_count+1)//2, len(self.frequencies_mhz), 4)
        integration_buffer = dsp.compy.zeros(correlation_shape, dtype="D")
        read_elapsed_s = 0.0
        concat_elapsed_s = 0.0
        transfer_elapsed_s = 0.0
        reorder_elapsed_s = 0.0
        t_progress = t
        while True:
            if data_spectra_count >= datablock_time_requirement:
                assert len(datablock.shape) == 5
                # enough data to process,
                # transfer to compute device
                if dsp.cupy_enabled:
                    t = time.perf_counter_ns()
                    datablock = dsp.compy.array(datablock)
                    transfer_elapsed_s += 1e-9*(time.perf_counter_ns() - t)

                # move Block to before T and merge dimensions
                if datablock.shape[0] == 1:
                    datablock = datablock.reshape(datablock.shape[1:])
                else:
                    t = time.perf_counter_ns()
                    datablock = dsp.compy.transpose(
                        datablock,
                        (1,2,0,3,4)
                    ).reshape((
                        datablock.shape[1],
                        datablock.shape[2],
                        datablock.shape[0]*datablock.shape[3],
                        datablock.shape[4]
                    ))
                    reorder_elapsed_s += 1e-9*(time.perf_counter_ns() - t)

                data_pos = self._inputhandler.data_bytes_processed()
                datasize_processed += data_pos - last_data_pos
                last_data_pos = data_pos
                progress = datasize_processed/self._inputhandler.data_bytes_total()

                t = time.perf_counter_ns()
                progress_elapsed_s = 1e-9*(t - t_progress)
                t_progress = t

                total_elapsed_s = 1e-9*(t - t_start)
                blri_logger.debug(f"Read time: {read_elapsed_s} s ({100*read_elapsed_s/progress_elapsed_s:0.2f} %)")
                blri_logger.debug(f"Concat time: {concat_elapsed_s} s ({100*concat_elapsed_s/progress_elapsed_s:0.2f} %)")
                blri_logger.debug(f"Transfer time: {transfer_elapsed_s} s ({100*transfer_elapsed_s/progress_elapsed_s:0.2f} %)")
                blri_logger.debug(f"Reorder time: {reorder_elapsed_s} s ({100*reorder_elapsed_s/progress_elapsed_s:0.2f} %)")
                blri_logger.debug(f"Running throughput: {datasize_processed/(total_elapsed_s*10**6):0.3f} MB/s")
                blri_logger.info(f"Progress: {datasize_processed/10**6:0.3f}/{self._inputhandler.data_bytes_total()/10**6:0.3f} MB ({100*progress:03.02f}%). Elapsed: {total_elapsed_s:0.3f} s, ETC: {total_elapsed_s*(1-progress)/progress:0.3f} s")

                read_elapsed_s = 0.0
                concat_elapsed_s = 0.0
                transfer_elapsed_s = 0.0
                reorder_elapsed_s = 0.0

                blri_logger.debug(f"Process steps in provided datablock: {datablock.shape[2]}/{datablock_time_requirement} = {datablock.shape[2]/datablock_time_requirement}")
                while datablock.shape[2] >= datablock_time_requirement:
                    assert len(datablock.shape) == 4
                    datablock_residual = datablock[:, :, datablock_time_requirement:, :]
                    datablock = datablock[:, :, 0:datablock_time_requirement, :]

                    datablock_bytesize = datablock.size * datablock.itemsize

                    t = time.perf_counter_ns()
                    datablock = dsp.upchannelise(
                        datablock[:, self.frequency_begin_coarseidx:self.frequency_end_coarseidx, :, :],
                        self.upchannelisation_rate
                    )[:, self.frequency_begin_fineidx:self.frequency_end_fineidx, :, :]

                    elapsed_s = 1e-9*(time.perf_counter_ns() - t)
                    blri_logger.debug(f"Channelisation: {datablock_bytesize/(elapsed_s*10**6)} MB/s")
                    datablock_bytesize = datablock.size * datablock.itemsize

                    t = time.perf_counter_ns()
                    datablock = dsp.correlate(datablock, conjugation_convention_flip=self.invert_correlation_conjugation)
                    elapsed_s = 1e-9*(time.perf_counter_ns() - t)
                    blri_logger.debug(f"Correlation: {datablock_bytesize/(elapsed_s*10**6)} MB/s")

                    t = time.perf_counter_ns()
                    assert datablock.shape[2] == 1
                    integration_buffer += datablock.reshape(integration_buffer.shape)
                    elapsed_s = 1e-9*(time.perf_counter_ns() - t)
                    blri_logger.debug(f"Integration {integration_count}/{self.integration_rate}: {datablock_bytesize/(elapsed_s*10**6)} MB/s")
                    integration_count += 1

                    datablock = datablock_residual
                    del datablock_residual

                    if integration_count < self.integration_rate:
                        continue

                    yield integration_buffer.get() if dsp.cupy_enabled else integration_buffer, julian_date_from_unix(
                            self._inputhandler.increment_time_taking_midpoint_unix(datablock_time_requirement*self.integration_rate)
                        )

                    integration_count = 0
                    integration_buffer.fill(0.0)

                data_spectra_count = datablock.shape[2]
                datablock = datablock.reshape([1, *datablock.shape])
                if dsp.cupy_enabled:
                    datablock = datablock.get()

            try:
                t = time.perf_counter_ns()
                inputdata = next(inputdata_iter)
                read_elapsed_s += 1e-9*(time.perf_counter_ns() - t)
                data_spectra_count += inputdata.shape[2]
                inputdata = inputdata.reshape([1, *inputdata.shape])
                if datablock.size == 0:
                    datablock = inputdata
                else:
                    t = time.perf_counter_ns()
                    datablock = numpy.concatenate(
                        (datablock, inputdata)
                    )
                    concat_elapsed_s += 1e-9*(time.perf_counter_ns() - t)
            except StopIteration:
                break



def correlate(
    inputhandler: Union[InputGuppiIterator, InputStampIterator],
    telescope_info_filepath: str,
    frequency_selection_center: Optional[float] = None, # decimal percentage
    frequency_selection_percentage: Optional[float] = None, # decimal percentage
    frequency_mhz_begin: Optional[float] = None,
    frequency_mhz_end: Optional[float] = None,
    upchannelisation_rate: int = 1,
    integration_rate: int = 1,
    invert_correlation_conjugation: Optional[bool] = False,
    invert_uvw_baselines: Optional[bool] = False,
    polarisations: Optional[str] = None,
    output_filepath: Optional[str] = None
) -> str: 
    telinfo = blri_telinfo.load_telescope_metadata(telescope_info_filepath)
    
    if output_filepath is None:
        output_filepath = inputhandler.output_filepath_default()
    else:
        output_filepath = output_filepath
    
    correlation_iter = CorrelationIterator(
        inputhandler,
        frequency_selection_center,
        frequency_selection_percentage,
        frequency_mhz_begin,
        frequency_mhz_end,
        upchannelisation_rate,
        integration_rate,
        invert_correlation_conjugation,
        invert_uvw_baselines,
        polarisations
    )

    metadata = correlation_iter.metadata

    if correlation_iter.metadata.antenna_names is not None:
        telinfo.antennas = blri_telinfo.filter_and_reorder_antenna_in_telinfo(
            telinfo,
            correlation_iter.metadata.antenna_names
        ).antennas
    assert len(telinfo.antennas) == correlation_iter.metadata.nof_antenna, f"len({telinfo.antennas}) != {correlation_iter.metadata.nof_antenna}"

    ant_1_array, ant_2_array = uvh5.get_uvh5_ant_arrays(telinfo.antennas)
    num_bls = len(ant_1_array)

    if correlation_iter.metadata.antenna_names is not None:
        telinfo.antennas = blri_telinfo.filter_and_reorder_antenna_in_telinfo(
            telinfo,
            correlation_iter.metadata.antenna_names
        ).antennas
    assert len(telinfo.antennas) == correlation_iter.metadata.nof_antenna, f"len({telinfo.antennas}) != {correlation_iter.metadata.nof_antenna}"

    ant_1_array, ant_2_array = uvh5.get_uvh5_ant_arrays(telinfo.antennas)
    num_bls = len(ant_1_array)

    assert len(correlation_iter.metadata.polarisation_chars) == correlation_iter.metadata.nof_polarisation
    polproducts = [
        f"{pol1}{pol2}"
        for pol1 in correlation_iter.metadata.polarisation_chars for pol2 in correlation_iter.metadata.polarisation_chars
    ]

    phase_center_radians = (
        correlation_iter.metadata.phase_center_rightascension_radians,
        correlation_iter.metadata.phase_center_declination_radians,
    )

    dut1 = correlation_iter.metadata.dut1_s

    jd_time_array = numpy.array((num_bls,), dtype='d')
    integration_time = numpy.array((num_bls,), dtype='d')
    integration_time.fill(upchannelisation_rate*integration_rate*correlation_iter.metadata.spectra_timespan_s)
    flags = numpy.zeros((num_bls, len(correlation_iter.frequencies_mhz), len(polproducts)), dtype='?')
    nsamples = numpy.ones(flags.shape, dtype='d')

    ant_xyz = numpy.array([ant.position for ant in telinfo.antennas])
    antenna_numbers = [ant.number for ant in telinfo.antennas]
    baseline_ant_1_indices = [antenna_numbers.index(antnum) for antnum in ant_1_array]
    baseline_ant_2_indices = [antenna_numbers.index(antnum) for antnum in ant_2_array]
    lla = (telinfo.longitude_radians, telinfo.latitude_radians, telinfo.altitude)
    
    datablock_time_requirement = upchannelisation_rate
    blri_logger.debug(f"Datablock time requirement per processing step: {datablock_time_requirement}")

    with h5py.File(output_filepath, "w") as f:
        uvh5_datasets = uvh5.uvh5_initialise(
            f,
            telinfo.telescope_name,
            correlation_iter.metadata.telescope,  # instrument_name
            correlation_iter.metadata.source_name,  # source_name,
            lla,
            telinfo.antennas,
            correlation_iter.frequencies_mhz*1e6,
            polproducts,
            phase_center_radians,
            dut1=dut1
        )
        
        for correlation_tuple in correlation_iter.data():
            correlation, time_jd = correlation_tuple
            t = time.perf_counter_ns()
            jd_time_array.fill(
                time_jd
            )

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
                    baseline_1_to_2=not invert_uvw_baselines
                ),
                jd_time_array,
                integration_time,
                correlation,
                flags,
                nsamples,
            )
            
            elapsed_s = 1e-9*(time.perf_counter_ns() - t)
            blri_logger.debug(f"Write: {(correlation.size * correlation.itemsize)/(elapsed_s*10**6)} MB/s")

    return output_filepath


def correlate_cli(arg_strs: list = None):
    parser = argparse.ArgumentParser(
        description="Correlate the data of a RAW file set, producing a UVH5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_filepaths",
        type=str,
        nargs="+",
        help="The path to the input files: GUPPI RAW file stem or all filepaths; Seticore Stamps filepath.",
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
        "--invert-uvw-baselines",
        action="store_true",
        help="Instead of baseline UVWs being ant1->ant2 (`ant2 subtract ant1`), set ant2->ant1 (`ant1 subtract ant2`)."
    )
    parser.add_argument(
        "--invert-correlation-conjugation",
        action="store_true",
        help="Instead of conjugating ant2, conjugate ant1 for correlation (effectively conjugating conventional correlation data)."
    )
    parser.add_argument(
        "--stamp-index",
        type=int,
        default=0,
        help="The selective index of a stamp within a '.stamps' file. Only applicable when `input_filepaths` is a Seticore Stamps filepath"
    )
    parser.add_argument(
        "--stamp-timestep-increment",
        type=int,
        default=0,
        help="Set the timestep increment of the iteration through the Stamp's data. Only applicable when `input_filepaths` is a Seticore Stamps filepath"
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
    
    if args.cupy:
        dsp.compute_with_cupy()
    else:
        dsp.compute_with_numpy()

    if os.path.splitext(args.input_filepaths[0])[1] == ".stamps":
        inputhandler = InputStampIterator(
            args.input_filepaths,
            stamp_index=args.stamp_index,
            timestep_increment=args.stamp_timestep_increment
        )
    else:
        inputhandler = InputGuppiIterator(
            args.input_filepaths,
            dtype=args.dtype,
            unsorted_raw_filepaths=args.unsorted_raw_filepaths
        )

    correlate(
        inputhandler,
        args.telescope_info_filepath,
        frequency_selection_center = args.frequency_selection_center,
        frequency_selection_percentage = args.frequency_selection_percentage,
        frequency_mhz_begin = args.frequency_mhz_begin,
        frequency_mhz_end = args.frequency_mhz_end,
        upchannelisation_rate = args.upchannelisation_rate,
        invert_correlation_conjugation = args.invert_correlation_conjugation,
        invert_uvw_baselines = args.invert_uvw_baselines,
        integration_rate = args.integration_rate,
        polarisations = args.polarisations,
    )

