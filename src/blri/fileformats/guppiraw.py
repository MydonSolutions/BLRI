import os
from typing import List

from pydantic import BaseModel
import numpy
from guppi import GuppiRawHandler

from blri import logger as blri_logger, parse
from blri.metadata.input import InputMetaData

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

    def metadata(self, polarisation_chars=None) -> InputMetaData:
        if polarisation_chars is None:
            polarisation_chars = self.guppi_header.get("POLS")
        assert polarisation_chars is not None
        nants, nchan, ntimes, npol = self.guppi_header.blockshape
        return InputMetaData(
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

