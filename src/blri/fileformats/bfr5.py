from typing import Dict, Tuple, List
from blri.fileformats.hdf5 import hdf5_fields_are_equal
from blri.fileformats.telinfo import AntennaDetail
from blri.times import julian_date_from_unix 

import numpy
import h5py

from blri.interferometry import delays

def write(
  output_filepath,
  obs_id: str,
  telescope_name: str,
  instrument_name: str,
  beam_src_coord_map: Dict[str, Tuple[float, float]],
  phase_center: Tuple[float, float],
  reference_lla: tuple, # (longitude:radians, latitude:radians, altitude)
  antennas: Dict[str, AntennaDetail],
  times_unix: numpy.ndarray,
  frequencies_hz: numpy.ndarray, # (nchan)
  calcoeff_bandpass: numpy.ndarray, # (nchan, npol, nants)
  calcoeff_gain: numpy.ndarray, # (npol, nants)
  dut1: float = 0.0,
  primary_center: Tuple[float, float] = None,
  reference_antenna_name: str = None
):
    nants = len(antennas)
    npol = calcoeff_gain.shape[0]
    nchan = len(frequencies_hz)
    nbeams = len(beam_src_coord_map)
    ntimes = len(times_unix)

    times_jd = julian_date_from_unix(times_unix)

    calcoeff = calcoeff_bandpass * numpy.reshape(numpy.repeat(calcoeff_gain, nchan, axis=0), (nchan, npol, nants))

    antenna_names = list(antennas.keys())
    if reference_antenna_name is None:
        reference_antenna_name = antenna_names[0]

    antenna_positions = numpy.array([ant.position for ant in antennas.values()])
    delay_ns = delays(
        antenna_positions,
        phase_center,
        numpy.array(list(beam_src_coord_map.values())),
        times_jd,
        reference_lla,
        reference_antenna_index = antenna_names.index(reference_antenna_name)
    )

    with h5py.File(output_filepath, "w") as f:
        dimInfo = f.create_group("diminfo")
        dimInfo.create_dataset("nants", data=nants)
        dimInfo.create_dataset("npol", data=npol)
        dimInfo.create_dataset("nchan", data=nchan)
        dimInfo.create_dataset("nbeams", data=nbeams)
        dimInfo.create_dataset("ntimes", data=ntimes)

        beamInfo = f.create_group("beaminfo")
        beamInfo.create_dataset("ras", data=numpy.array([beam[0] for beam in beam_src_coord_map.values()]), dtype='d') # radians
        beamInfo.create_dataset("decs", data=numpy.array([beam[1] for beam in beam_src_coord_map.values()]), dtype='d') # radians
        source_names = [beam.encode() for beam in beam_src_coord_map.keys()]
        longest_source_name = max(len(name) for name in source_names)
        beamInfo.create_dataset("src_names", data=numpy.array(source_names, dtype=f"S{longest_source_name}"), dtype=h5py.special_dtype(vlen=str))

        calInfo = f.create_group("calinfo")
        calInfo.create_dataset("refant", data=reference_antenna_name.encode())
        calInfo.create_dataset("cal_K", data=numpy.ones((npol, nants)), dtype='d')
        calInfo.create_dataset("cal_B", data=calcoeff_bandpass, dtype='D')
        calInfo.create_dataset("cal_G", data=calcoeff_gain, dtype='D')
        calInfo.create_dataset("cal_all", data=calcoeff, dtype='D')

        delayInfo = f.create_group("delayinfo")
        delayInfo.create_dataset("delays", data=delay_ns, dtype='d')
        delayInfo.create_dataset("rates", data=numpy.zeros((ntimes, nbeams, nants)), dtype='d')
        delayInfo.create_dataset("time_array", data=times_unix, dtype='d')
        delayInfo.create_dataset("jds", data=times_jd, dtype='d')
        delayInfo.create_dataset("dut1", data=dut1, dtype='d')

        obsInfo = f.create_group("obsinfo")
        obsInfo.create_dataset("obsid", data=obs_id.encode())
        obsInfo.create_dataset("freq_array", data=frequencies_hz*1e-9, dtype='d') # GHz
        obsInfo.create_dataset("phase_center_ra", data=phase_center[0], dtype='d') # radians
        obsInfo.create_dataset("phase_center_dec", data=phase_center[1], dtype='d') # radians
        if primary_center is not None:
            obsInfo.create_dataset("primary_center_ra", data=primary_center[0], dtype='d') # radians
            obsInfo.create_dataset("primary_center_dec", data=primary_center[1], dtype='d') # radians
        obsInfo.create_dataset("instrument_name", data=instrument_name.encode())

        telInfo = f.create_group("telinfo")
        telInfo.create_dataset("antenna_positions", data=antenna_positions, dtype='d')
        telInfo.create_dataset("antenna_position_frame", data="xyz".encode())
        longest_antenna_name = max(*[len(name) for name in antenna_names])
        telInfo.create_dataset("antenna_names", data=numpy.array(antenna_names, dtype=f"S{longest_antenna_name}"), dtype=h5py.special_dtype(vlen=str))
        telInfo.create_dataset("antenna_numbers", data=numpy.array([ant.number for ant in antennas.values()]), dtype='i')
        telInfo.create_dataset("antenna_diameters", data=numpy.array([ant.diameter for ant in antennas.values()]), dtype='d')
        telInfo.create_dataset("longitude", data=reference_lla[0]*180/numpy.pi, dtype='d') # degrees
        telInfo.create_dataset("latitude", data=reference_lla[1]*180/numpy.pi, dtype='d') # degrees
        telInfo.create_dataset("altitude", data=reference_lla[2], dtype='d')
        telInfo.create_dataset("telescope_name", data=telescope_name.encode())


def bfr5_differences(filepath_a: str, filepath_b: str, atol: float=1e-8, rtol: float=1e-5):
    with h5py.File(filepath_a, 'r') as h5_a:
        with h5py.File(filepath_b, 'r') as h5_b:
            return {
                group: [
                    field
                    for field in h5_a[group]
                    if not hdf5_fields_are_equal(
                        h5_a[group][field],
                        h5_b[group][field]
                    )
                ]
                
                for group in [
                    "diminfo",
                    "beaminfo",
                    "calinfo",
                    "delayinfo",
                    "obsinfo",
                    "telinfo",
                ]
            }
