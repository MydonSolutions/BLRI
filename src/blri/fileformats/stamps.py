import os
from typing import List

from pydantic import BaseModel
import numpy

from blri import logger as blri_logger
from blri.metadata.input import InputMetaData

# TODO InputIterators provide datablock_time_requirement always


class InputStampIterator:
    class StampTimeKeeper(BaseModel):
        spectra_timespan_s: float
        time_unix_offset: float
        running_time_unix: float

    def __init__(self, stamp_filepaths: List[str], stamp_index=0, timestep_increment=0):
        from seticore import stamp_capnp # TODO better handle optional package dependency

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
        self.timestep_increment = timestep_increment if timestep_increment > 0 else self.stamp.numTimesteps
        assert self.stamp.numTimesteps % self.timestep_increment == 0, f"{self.stamp.numTimesteps} % {self.timestep_increment} = {self.stamp.numTimesteps % self.timestep_increment} != 0"

        self._data_bytes_processed = 0
        blri_logger.debug(f"""Stamp data shape (T, F, P, A): {(
            self.stamp.numTimesteps,
            self.stamp.numChannels,
            self.stamp.numPolarizations,
            self.stamp.numAntennas,
        )}""")
        blri_logger.debug(f"{self.stamp.numTimesteps // self.timestep_increment} steps in time, each spanning {self.timestep_increment}.")

        self.stamp_bytes_total = numpy.prod((
            self.stamp.numTimesteps,
            self.stamp.numChannels,
            self.stamp.numPolarizations,
            self.stamp.numAntennas,
            2, # 2 components in complexity
            4 # 4 bytes in Float32
        ))
        blri_logger.debug(f"Total Stamp Data bytes: {self.stamp_bytes_total/(10**6)} MB")

    def metadata(self, polarisation_chars=None) -> InputMetaData:
        assert polarisation_chars is not None
        return InputMetaData(
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
        self._data_bytes_processed = 0
        all_data = numpy.transpose(numpy.array(
                self.stamp.data
            ).view(
                numpy.complex128 # python automatically upscaled the float32
            ).reshape(
                data_shape
            ),
            (3,1,0,2)
        )
        
        for timestep_index in range(
            0,
            self.stamp.numTimesteps,
            self.timestep_increment
        ):
            self._data_bytes_processed += (self.stamp_bytes_total//self.stamp.numTimesteps) * self.timestep_increment 
            yield all_data[
                :,
                :,
                timestep_index:timestep_index+self.timestep_increment,
                :
            ]

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

