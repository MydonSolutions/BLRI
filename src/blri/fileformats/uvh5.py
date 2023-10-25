from typing import List
from dataclasses import dataclass

import numpy
import h5py
import erfa

from blri.fileformats.telinfo import AntennaDetail
from blri import __version__


POLARISATION_MAP = {
  "i" :  1, "q" :  2, "u" :  3,  "v":  4,
  "rr": -1, "ll": -2, "rl": -3, "lr": -4,
  "xx": -5, "yy": -6, "xy": -7, "yx": -8,
}


def get_uvh5_ant_arrays(
    antennas: List[AntennaDetail],
):
    ant_1_array = []
    ant_2_array = []
    # auto correlations first
    for ant in antennas:
        ant_1_array.append(ant.number)
        ant_2_array.append(ant.number)

    # cross correlations
    for ant_i, ant1 in enumerate(antennas):
        for ant2 in antennas[ant_i+1:]:
            ant_1_array.append(ant1.number)
            ant_2_array.append(ant2.number)

    return ant_1_array, ant_2_array


def get_polarisation_array(polarisation_strings: list):
    return list(map(
        lambda polstr: POLARISATION_MAP[polstr.lower()],
        polarisation_strings
    ))


def get_uvw_array(
    time_jd,
    source_radec_radians,
    ant_coordinates,
    lla,
    baseline_ant_1_indices,
    baseline_ant_2_indices,
    dut1=0.0,
):
    """Computes UVW antenna coordinates with respect to reference position. There-after constructs baseline relative-UVWs array.

    Args:
        time_jd: Julian Date time of pointing
        source: (RA, Dec) tuple
        ant_coordinates: numpy.ndarray
            Antenna XYZ coordinates, relative to reference position. This is indexed as (antenna_number, xyz)
        lla: tuple Reference Coordinates (radians)
            Longitude, Latitude, Altitude. The antenna_coordinates must have
            this component in them.

    Returns:
        The baseline UVW coordinates.
    """

    aob, zob, ha_rad, dec_rad, rob, eo = erfa.atco13(
        *source_radec_radians,
        0, 0, 0, 0,
        time_jd, 0,
        dut1,
        *lla,
        0, 0,
        0, 0, 0, 0
    )

    sin_long_minus_hangle = numpy.sin(lla[0]-ha_rad)
    cos_long_minus_hangle = numpy.cos(lla[0]-ha_rad)
    sin_declination = numpy.sin(dec_rad)
    cos_declination = numpy.cos(dec_rad)

    uvws = numpy.zeros(ant_coordinates.shape, dtype=numpy.float64)

    for ant in range(ant_coordinates.shape[0]):
        # RotZ(long-ha) anti-clockwise
        x = cos_long_minus_hangle*ant_coordinates[ant, 0] - (-sin_long_minus_hangle)*ant_coordinates[ant, 1]
        y = (-sin_long_minus_hangle)*ant_coordinates[ant, 0] + cos_long_minus_hangle*ant_coordinates[ant, 1]
        z = ant_coordinates[ant, 2]

        # RotY(declination) clockwise
        x_ = x
        x = cos_declination*x_ + sin_declination*z
        z = -sin_declination*x_ + cos_declination*z

        # Permute (WUV) to (UVW)
        uvws[ant, 0] = y
        uvws[ant, 1] = z
        uvws[ant, 2] = x

    return numpy.array([ # ant_1 -> ant_2
        uvws[baseline_ant_2_indices[baseline_i], :] - uvws[baseline_ant_1_indices[baseline_i], :]
        for baseline_i in range(len(baseline_ant_1_indices))
    ])


@dataclass
class Uvh5DynamicDatasets:
    header_ntimes: h5py.Dataset
    header_nblts: h5py.Dataset
    header_ant_1_array: h5py.Dataset
    header_ant_2_array: h5py.Dataset
    header_uvw_array: h5py.Dataset
    header_time_array: h5py.Dataset
    header_integration_time: h5py.Dataset
    data_visdata: h5py.Dataset
    data_flags: h5py.Dataset
    data_nsamples: h5py.Dataset


def uvh5_initialise(
    fout: h5py.File,
    telescope_name: str,
    instrument_name: str,
    source_name: str,
    reference_lla: tuple, # (longitude:radians, latitude:radians, altitude)
    antennas: List[AntennaDetail],
    frequencies_hz: numpy.ndarray, # (nchan)
    polproducts: List[str], # (npol*npol)
    phase_center_ra_dec_radians_icrs2000: tuple,
    dut1: float = 0.0
) -> Uvh5DynamicDatasets:

    num_bls = len(antennas)*(len(antennas)+1)//2
    num_freqs = len(frequencies_hz)
    num_polprods = len(polproducts)

    uvh5g_header = fout.create_group("Header")
    uvh5g_data = fout.create_group("Data")

    uvh5g_header.create_dataset("longitude", data=reference_lla[0]*180/numpy.pi, dtype='d') # degrees
    uvh5g_header.create_dataset("latitude", data=reference_lla[1]*180/numpy.pi, dtype='d') # degrees
    uvh5g_header.create_dataset("altitude", data=reference_lla[2], dtype='d')

    uvh5g_header.create_dataset("telescope_name", data=telescope_name.encode())
    uvh5g_header.create_dataset("instrument", data=instrument_name.encode())
    uvh5g_header.create_dataset("object_name", data=source_name.encode())
    uvh5g_header.create_dataset("history", data=f"github.com/MydonSolutions/BLRI v{__version__}")
    uvh5g_header.create_dataset("phase_type", data="phased")
    uvh5g_header.create_dataset("Nants_data", data=len(antennas))
    uvh5g_header.create_dataset("Nants_telescope", data=len(antennas))

    antenna_names = [ant.name for ant in antennas]
    uvh5g_header.create_dataset("antenna_names", data=numpy.array(antenna_names, dtype=f"S{max(map(len, antenna_names))}"), dtype=h5py.special_dtype(vlen=str))
    uvh5g_header.create_dataset("antenna_numbers", data=numpy.array([ant.number for ant in antennas]), dtype='i')
    uvh5g_header.create_dataset("antenna_diameters", data=numpy.array([ant.diameter for ant in antennas]), dtype='d')
    uvh5g_header.create_dataset("antenna_positions", data=numpy.array([ant.position for ant in antennas]), dtype='d')

    uvh5g_header.create_dataset("Nbls", data=num_bls)
    uvh5g_header.create_dataset("Nfreqs", data=num_freqs)
    uvh5g_header.create_dataset("Npols", data=num_polprods)
    uvh5g_header.create_dataset("freq_array", data=frequencies_hz, dtype='d')
    channel_width = [frequencies_hz[i+1]-frequencies_hz[i] for i in range(len(frequencies_hz)-1)]
    channel_width.append(channel_width[-1])
    assert len(channel_width) == len(frequencies_hz)
    uvh5g_header.create_dataset("channel_width", data=numpy.array(channel_width), dtype='d')
    uvh5g_header.create_dataset("Nspws", data=1)
    uvh5g_header.create_dataset("spw_array", data=numpy.ones((1), dtype='i'))
    uvh5g_header.create_dataset("flex_spw", data=False)

    uvh5g_header.create_dataset("polarization_array", data=numpy.array(get_polarisation_array(polproducts)), dtype='i')

    uvh5g_header.create_dataset("version", data="1.0".encode())
    # uvh5g_header.create_dataset("flex_spw_id_array", data=) # 1 int
    uvh5g_header.create_dataset("dut1", data=dut1, dtype='d')
    # uvh5g_header.create_dataset("earth_omega", data=) # 0 double
    # uvh5g_header.create_dataset("gst0", data=) # 0 double
    # uvh5g_header.create_dataset("rdate", data=) # 0 string
    # uvh5g_header.create_dataset("timesys", data=) # 0 string
    # uvh5g_header.create_dataset("x_orientation", data=) # 0 string
    # uvh5g_header.create_dataset("uvplane_reference_time", data=) # 0 int
    uvh5g_header.create_dataset("phase_center_ra", data=phase_center_ra_dec_radians_icrs2000[0], dtype='d')
    uvh5g_header.create_dataset("phase_center_dec", data=phase_center_ra_dec_radians_icrs2000[1], dtype='d')
    uvh5g_header.create_dataset("phase_center_epoch", data=2000.0)
    uvh5g_header.create_dataset("phase_center_frame", data="icrs".encode())

    return Uvh5DynamicDatasets(
        header_ntimes=uvh5g_header.create_dataset("Ntimes", data=0),
        header_nblts=uvh5g_header.create_dataset("Nblts", data=0),
        header_ant_1_array=uvh5g_header.create_dataset("ant_1_array", (0,), dtype='i', maxshape=(None,)),
        header_ant_2_array=uvh5g_header.create_dataset("ant_2_array", (0,), dtype='i', maxshape=(None,)),
        header_uvw_array=uvh5g_header.create_dataset("uvw_array", (0, 3), dtype='d', maxshape=(None, 3)),
        header_time_array=uvh5g_header.create_dataset("time_array", (0,), dtype='d', maxshape=(None,)),
        header_integration_time=uvh5g_header.create_dataset("integration_time", (0,), dtype='d', maxshape=(None,)),

        data_visdata=uvh5g_data.create_dataset("visdata", (0, num_freqs, num_polprods), dtype='D', maxshape=(None, num_freqs, num_polprods)),
        data_flags=uvh5g_data.create_dataset("flags", (0, num_freqs, num_polprods), dtype='?', maxshape=(None, num_freqs, num_polprods)),
        data_nsamples=uvh5g_data.create_dataset("nsamples", (0, num_freqs, num_polprods), dtype='d', maxshape=(None, num_freqs, num_polprods)),
    )


def uvh5_write_chunk(
    uvh5_datasets: Uvh5DynamicDatasets,
    ant_1_array,
    ant_2_array,
    uvw_array,
    jd_time_array,
    integration_time,
    visdata,
    flags,
    nsamples,
):
    num_bls, num_freqs, num_polprods = visdata.shape

    uvh5_datasets.header_ntimes[()] += 1
    uvh5_datasets.header_nblts[()] += num_bls
    num_bltimes = uvh5_datasets.header_nblts[()]


    uvh5_datasets.header_ant_1_array.resize((num_bltimes,))
    uvh5_datasets.header_ant_1_array[-num_bls:] = ant_1_array

    uvh5_datasets.header_ant_2_array.resize((num_bltimes,))
    uvh5_datasets.header_ant_2_array[-num_bls:] = ant_2_array

    uvh5_datasets.header_uvw_array.resize((num_bltimes, 3))
    uvh5_datasets.header_uvw_array[-num_bls:, :] = uvw_array

    uvh5_datasets.header_time_array.resize((num_bltimes,))
    uvh5_datasets.header_time_array[-num_bls:] = jd_time_array

    uvh5_datasets.header_integration_time.resize((num_bltimes,))
    uvh5_datasets.header_integration_time[-num_bls:] = integration_time

    uvh5_datasets.data_visdata.resize((num_bltimes, num_freqs, num_polprods))
    uvh5_datasets.data_visdata[-num_bls:, :, :] = visdata

    uvh5_datasets.data_flags.resize((num_bltimes, num_freqs, num_polprods))
    uvh5_datasets.data_flags[-num_bls:, :, :] = flags

    uvh5_datasets.data_nsamples.resize((num_bltimes, num_freqs, num_polprods))
    uvh5_datasets.data_nsamples[-num_bls:, :, :] = nsamples


def _uvh5_field_is_equal(
        h5_a_field,
        h5_b_field,
    ):
    try:
        if h5_a_field.shape == tuple():
            return h5_a_field[()] == h5_b_field[()]
        else:
            return (h5_a_field[:] == h5_b_field[:]).all()
    except ValueError:
        return False


def uvh5_differences(filepath_a: str, filepath_b: str, atol: float=1e-8, rtol: float=1e-5):
    with h5py.File(filepath_a, 'r') as h5_a:
        with h5py.File(filepath_b, 'r') as h5_b:
            header_differences = [
                field
                for field in h5_a["Header"]
                if not _uvh5_field_is_equal(
                    h5_a["Header"][field],
                    h5_b["Header"][field]
                )
            ]
            data_differences = [
                field
                for field in [
                    "flags",
                    "nsamples"
                ]
                if not _uvh5_field_is_equal(
                    h5_a["Data"][field],
                    h5_b["Data"][field]
                )
            ]
            visdata_is_equal = False
            try:
                visdata_is_equal = numpy.allclose(
                    h5_a["Data"]["visdata"],
                    h5_b["Data"]["visdata"],
                    atol=atol,
                    rtol=rtol
                )
            except ValueError:
                pass

            if not visdata_is_equal:
                data_differences.append("visdata")
            
            return header_differences, data_differences
