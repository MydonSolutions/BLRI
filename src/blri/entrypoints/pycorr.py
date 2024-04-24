import os, argparse, logging, time
from typing import Optional, List

from pydantic import BaseModel
import h5py
import numpy
from guppi import GuppiRawHandler

from blri import logger as blri_logger, parse, dsp
from blri.fileformats import telinfo as blri_telinfo, uvh5
from blri.times import julian_date_from_unix


class MetaData(BaseModel):
    nof_antenna: int
    nof_channel: int
    nof_time: int
    nof_polarisation: int
    channel_bandwidth_mhz: float
    observed_frequency_mhz: float
    polarisation_chars: str
    phase_center_rightascension_radians: float
    phase_center_declination_radians: float
    dut1_s: float
    spectra_timespan_s: float
    telescope: str
    source_name: str
    antenna_names: Optional[List[str]]


# TODO InputIterators provide datablock_time_requirement always
    
class InputGuppiIterator:
    class GuppiTimeKeeper(BaseModel):
        nof_spectra_per_block: int
        nof_packet_indices_per_block: int
        sample_packet_index: int
        spectra_timespan_s: float
        time_unix_offset: float
    
    def __init__(self, raw_filepaths: List[str], dtype="float32", unsorted_raw_filepaths=False):
        if len(raw_filepaths) == 1 and not os.path.exists(raw_filepaths[0]):
            # argument appears to be a singular stem, break it out of the list
            raw_filepaths = raw_filepaths[0]
        elif not unsorted_raw_filepaths:
            raw_filepaths.sort()

        self.guppi_handler = GuppiRawHandler(raw_filepaths)
        self.guppi_blocks_iter = self.guppi_handler.blocks(astype=numpy.dtype(dtype).type)
        self.guppi_header, self.current_guppi_data = next(self.guppi_blocks_iter)
        self.timekeeper = self.GuppiTimeKeeper(
            nof_spectra_per_block = self.guppi_header.nof_spectra_per_block,
            nof_packet_indices_per_block = self.guppi_header.nof_packet_indices_per_block,
            sample_packet_index = self.guppi_header.packet_index,
            spectra_timespan_s = self.guppi_header.spectra_timespan,
            time_unix_offset = self.guppi_header.time_unix_offset
        )

        self._data_bytes_processed = 1

        blri_logger.debug(f"GUPPI Block shape (A, F, T, P): {self.guppi_header.blockshape}")
        blri_logger.debug(f"GUPPI RAW files: {self.guppi_handler._guppi_filepaths}")
        self.guppi_bytes_total = numpy.sum(list(map(os.path.getsize, self.guppi_handler._guppi_filepaths)))
        blri_logger.debug(f"Total GUPPI RAW bytes: {self.guppi_bytes_total/(10**6)} MB")
    
    def metadata(self, polarisation_chars=None) -> MetaData:
        if polarisation_chars is None:
            polarisation_chars = self.guppi_header.get("POLS")
        assert polarisation_chars is not None
        nants, nchan, ntimes, npol = self.guppi_header.blockshape
        return MetaData(            
            nof_antenna = nants,
            nof_channel = nchan,
            nof_time = ntimes,
            nof_polarisation = npol,
            channel_bandwidth_mhz = self.guppi_header.channel_bandwidth,
            observed_frequency_mhz = self.guppi_header.observed_frequency,
            polarisation_chars = polarisation_chars,
            phase_center_rightascension_radians = parse.degrees_process(self.guppi_header.get("RA_PHAS", self.guppi_header.rightascension_string)) * numpy.pi / 12.0,
            phase_center_declination_radians = parse.degrees_process(self.guppi_header.get("DEC_PHAS", self.guppi_header.declination_string)) * numpy.pi / 180.0,
            dut1_s = self.guppi_header.get("DUT1", 0.0),
            spectra_timespan_s = self.guppi_header.spectra_timespan,
            telescope = self.guppi_header.telescope,
            source_name = self.guppi_header.source_name,
            antenna_names = self.guppi_header.antenna_names if hasattr(self.guppi_header, "antenna_names") else None
        )
    
    def data(self):
        self._data_bytes_processed = self.guppi_handler._guppi_file_handle.tell()
        while self.current_guppi_data is not None:
            yield self.current_guppi_data

            prev_guppi_file_index = self.guppi_handler._guppi_file_index
            prev_pos = self.guppi_handler._guppi_file_handle.tell()

            try:
                _, self.current_guppi_data = next(self.guppi_blocks_iter)
            except StopIteration:
                self.current_guppi_data = None
                break

            if self.guppi_handler._guppi_file_index != prev_guppi_file_index:
                prev_pos = 0
            self._data_bytes_processed += self.guppi_handler._guppi_file_handle.tell() - prev_pos

    def increment_time_taking_midpoint_unix(self, step_timesamples) -> float:
        packet_index_step = step_timesamples*self.timekeeper.nof_packet_indices_per_block/self.timekeeper.nof_spectra_per_block

        unix_midpoint = self.timekeeper.time_unix_offset + (
            (self.timekeeper.sample_packet_index + packet_index_step/2)
            * (self.timekeeper.nof_spectra_per_block/self.timekeeper.nof_packet_indices_per_block)
        ) * self.timekeeper.spectra_timespan_s

        self.timekeeper.sample_packet_index += packet_index_step
        return unix_midpoint

    def output_filepath_default(self) -> str:
        input_dir, input_filename = os.path.split(self.guppi_handler._guppi_filepaths[0])
        return os.path.join(input_dir, f"{os.path.splitext(input_filename)[0]}.uvh5")
    
    def data_bytes_total(self) -> int:
        return self.guppi_bytes_total
    
    def data_bytes_processed(self) -> int:
        return self._data_bytes_processed


class InputStampIterator:
    class StampTimeKeeper(BaseModel):
        spectra_timespan_s: float
        time_unix_offset: float
        running_time_unix: float
    
    def __init__(self, stamp_filepaths: List[str], stamp_index=0):
        from seticore import stamp_capnp

        self.stamp = None
        self.stamp_filepath = stamp_filepaths[0]
        with open(self.stamp_filepath) as f:
            stamps = stamp_capnp.Stamp.read_multiple(f, traversal_limit_in_words=2**30)
            for i, stamp in enumerate(stamps):
                if i == stamp_index:
                    self.stamp = stamp
                    break
        if self.stamp is None:
            raise RuntimeError(f"Did not reach index {stamp_index} within {self.stamp_filepath}")

        self.timekeeper = self.StampTimeKeeper(
            spectra_timespan_s = stamp.tsamp,
            time_unix_offset = stamp.tstart,
            running_time_unix = 0
        )

        self._data_bytes_processed = 0
        blri_logger.debug(f"""Stamp data shape (T, F, P, A): {(
            self.stamp.numTimesteps,
            self.stamp.numChannels,
            self.stamp.numPolarizations,
            self.stamp.numAntennas,
        )}""")

        self.stamp_bytes_total = numpy.prod((
            self.stamp.numTimesteps,
            self.stamp.numChannels,
            self.stamp.numPolarizations,
            self.stamp.numAntennas,
            2, # 2 components in complexity
            4 # 4 bytes in Float32
        ))
        blri_logger.debug(f"Total Stamp Data bytes: {self.stamp_bytes_total/(10**6)} MB")
    
    def metadata(self, polarisation_chars=None) -> MetaData:
        assert polarisation_chars is not None
        return MetaData(
            nof_antenna = self.stamp.numAntennas,
            nof_channel = self.stamp.numChannels,
            nof_time = self.stamp.numTimesteps,
            nof_polarisation = self.stamp.numPolarizations,
            channel_bandwidth_mhz = self.stamp.foff,
            observed_frequency_mhz = self.stamp.fch1 + (self.stamp.numChannels/2)*self.stamp.foff,
            polarisation_chars = polarisation_chars,
            phase_center_rightascension_radians = self.stamp.ra*numpy.pi/12,
            phase_center_declination_radians = self.stamp.dec*numpy.pi/180,
            dut1_s = 0.0,
            spectra_timespan_s = self.stamp.tsamp,
            telescope = f"{self.stamp.telescopeId}",
            source_name = self.stamp.sourceName,
            antenna_names = None
        )
    
    def data(self):
        data_shape = (
            self.stamp.numTimesteps,
            self.stamp.numChannels,
            self.stamp.numPolarizations,
            self.stamp.numAntennas,
        )
        data = numpy.array(
            self.stamp.data
        ).view(
            numpy.complex128 # python automatically upscaled the float32
        ).reshape(
            data_shape
        )

        self._data_bytes_processed = self.stamp_bytes_total
        yield numpy.transpose(
            data,
            (3,1,0,2)
        )

    def increment_time_taking_midpoint_unix(self, step_timesamples) -> float:
        time_step = step_timesamples*self.timekeeper.spectra_timespan_s
        unix_midpoint = self.timekeeper.time_unix_offset + self.timekeeper.running_time_unix + time_step/2
        self.timekeeper.running_time_unix += time_step
        return unix_midpoint

    def output_filepath_default(self) -> str:
        input_dir, input_filename = os.path.split(self.stamp_filepath)
        return os.path.join(input_dir, f"{os.path.splitext(input_filename)[0]}.uvh5")
    
    def data_bytes_total(self) -> int:
        return self.stamp_bytes_total
    
    def data_bytes_processed(self) -> int:
        return self._data_bytes_processed


def main(arg_strs: list = None):
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

    datablock_time_requirement = args.upchannelisation_rate
    blri_logger.debug(f"Datablock time requirement per processing step: {datablock_time_requirement}")

    telinfo = blri_telinfo.load_telescope_metadata(args.telescope_info_filepath)
    
    if os.path.splitext(args.input_filepaths[0])[1] == ".stamps":
        inputhandler = InputStampIterator(
            args.input_filepaths,
            stamp_index=args.stamp_index
        )
    else:
        inputhandler = InputGuppiIterator(
            args.input_filepaths,
            dtype=args.dtype,
            unsorted_raw_filepaths=args.unsorted_raw_filepaths
        )
    inputdata_iter = inputhandler.data()
    inputdata = next(inputdata_iter) # premature check to ensure data can be accessed

    if args.output_filepath is None:
        output_filepath = inputhandler.output_filepath_default()
    else:
        output_filepath = args.output_filepath
    
    metadata = inputhandler.metadata(polarisation_chars=args.polarisations)

    if metadata.antenna_names is not None:
        telinfo.antennas = blri_telinfo.filter_and_reorder_antenna_in_telinfo(
            telinfo,
            metadata.antenna_names
        ).antennas
    assert len(telinfo.antennas) == metadata.nof_antenna, f"len({telinfo.antennas}) != {metadata.nof_antenna}"

    ant_1_array, ant_2_array = uvh5.get_uvh5_ant_arrays(telinfo.antennas)
    num_bls = len(ant_1_array)

    observed_frequency_bottom_mhz = metadata.observed_frequency_mhz - (metadata.nof_channel/2)*metadata.channel_bandwidth_mhz

    upchan_bw = metadata.channel_bandwidth_mhz/args.upchannelisation_rate
    # offset to center of channels is deferred
    frequencies_mhz = observed_frequency_bottom_mhz + numpy.arange(metadata.nof_channel*args.upchannelisation_rate)*upchan_bw

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
        if frequency_end_coarseidx <= metadata.nof_channel:
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
    # offset to center of channels
    frequencies_mhz += 0.5 * upchan_bw

    assert len(metadata.polarisation_chars) == metadata.nof_polarisation
    polproducts = [
        f"{pol1}{pol2}"
        for pol1 in metadata.polarisation_chars for pol2 in metadata.polarisation_chars
    ]

    phase_center_radians = (
        metadata.phase_center_rightascension_radians,
        metadata.phase_center_declination_radians,
    )

    dut1 = metadata.dut1_s

    jd_time_array = numpy.array((num_bls,), dtype='d')
    integration_time = numpy.array((num_bls,), dtype='d')
    integration_time.fill(args.upchannelisation_rate*args.integration_rate*metadata.spectra_timespan_s)
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
            metadata.telescope,  # instrument_name
            metadata.source_name,  # source_name,
            lla,
            telinfo.antennas,
            frequencies_mhz*1e6,
            polproducts,
            phase_center_radians,
            dut1=dut1
        )

        datablock_shape = list(inputdata.shape)
        datablocks_per_requirement = datablock_time_requirement/datablock_shape[2]
        blri_logger.debug(f"Collects ceil({datablocks_per_requirement}) blocks for correlation.")
        datablock_shape[2] = numpy.ceil(datablocks_per_requirement)*datablock_shape[2]

        data_spectra_count = inputdata.shape[2]
        # while gathering blocks of data, `datablock` is shape (Block, A, F, T, P)
        datablock = inputdata.reshape([1, *inputdata.shape])

        t = time.perf_counter_ns()
        t_start = t
        last_data_pos = 0
        datasize_processed = 0
        integration_count = 0
        # Integrate fine spectra in a separate buffer
        integration_buffer = dsp.compy.zeros(flags.shape, dtype="D")
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

                data_pos = inputhandler.data_bytes_processed()
                datasize_processed += data_pos - last_data_pos
                last_data_pos = data_pos
                progress = datasize_processed/inputhandler.data_bytes_total()

                t = time.perf_counter_ns()
                progress_elapsed_s = 1e-9*(t - t_progress)
                t_progress = t

                total_elapsed_s = 1e-9*(t - t_start)
                blri_logger.debug(f"Read time: {read_elapsed_s} s ({100*read_elapsed_s/progress_elapsed_s:0.2f} %)")
                blri_logger.debug(f"Concat time: {concat_elapsed_s} s ({100*concat_elapsed_s/progress_elapsed_s:0.2f} %)")
                blri_logger.debug(f"Transfer time: {transfer_elapsed_s} s ({100*transfer_elapsed_s/progress_elapsed_s:0.2f} %)")
                blri_logger.debug(f"Reorder time: {reorder_elapsed_s} s ({100*reorder_elapsed_s/progress_elapsed_s:0.2f} %)")
                blri_logger.debug(f"Running throughput: {datasize_processed/(total_elapsed_s*10**6):0.3f} MB/s")
                blri_logger.info(f"Progress: {datasize_processed/10**6:0.3f}/{inputhandler.data_bytes_total()/10**6:0.3f} MB ({100*progress:03.02f}%). Elapsed: {total_elapsed_s:0.3f} s, ETC: {total_elapsed_s*(1-progress)/progress:0.3f} s")

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
                        datablock[:, frequency_begin_coarseidx:frequency_end_coarseidx, :, :],
                        args.upchannelisation_rate
                    )[:, frequency_begin_fineidx:frequency_end_fineidx, :, :]

                    elapsed_s = 1e-9*(time.perf_counter_ns() - t)
                    blri_logger.debug(f"Channelisation: {datablock_bytesize/(elapsed_s*10**6)} MB/s")
                    datablock_bytesize = datablock.size * datablock.itemsize

                    t = time.perf_counter_ns()
                    datablock = dsp.correlate(datablock, conjugation_convention_flip=args.invert_correlation_conjugation)
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
                    jd_time_array.fill(
                        julian_date_from_unix(
                            inputhandler.increment_time_taking_midpoint_unix(datablock_time_requirement*args.integration_rate)
                        )
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
                            baseline_1_to_2=not args.invert_uvw_baselines
                        ),
                        jd_time_array,
                        integration_time,
                        integration_buffer.get() if dsp.cupy_enabled else integration_buffer,
                        flags,
                        nsamples,
                    )
                    elapsed_s = 1e-9*(time.perf_counter_ns() - t)
                    blri_logger.debug(f"Write: {datablock_bytesize/(elapsed_s*10**6)} MB/s")

                    integration_count = 0
                    integration_buffer.fill(0.0)

                    data_spectra_count = datablock.shape[2]
                    datablock = datablock.reshape([1, *datablock.shape])
                    if dsp.cupy_enabled:
                        datablock = datablock.get()
                    break

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

    return output_filepath
