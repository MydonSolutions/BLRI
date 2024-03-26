from typing import List, Tuple

import numpy
import erfa

from blri.coords import compute_uvw_from_xyz


def delays(
    antenna_positions_xyz: numpy.ndarray, # [Antenna, XYZ] relative to whatever reference position
    boresight_radec_rad: Tuple[float, float],
    beam_radec_rad: List[Tuple[float, float]],
    times_jd: numpy.ndarray, # [julian_date]
    longlatalt_rad: Tuple[float, float, float], # Longitude, Latitude, Altitude (radians)
    reference_antenna_index: int = 0,
    dut1: float = 0.0
):
    """
    Calculate the nanosecond delays between antenna, for each beam, for each time.

    Return
    ------
        delays_ns (T, B, A)
    """

    delays_ns = numpy.zeros(
        (
            times_jd.shape[0],
            len(beam_radec_rad),
            antenna_positions_xyz.shape[0],
        ),
        dtype=numpy.float64
    )

    for t, tjd in enumerate(times_jd):

        # get valid eraASTROM instance
        astrom, eo = erfa.apco13(
            tjd, 0,
            dut1,
            *longlatalt_rad,
            0, 0,
            0, 0, 0, 0
        )

        boresightUvw = compute_uvw_from_xyz(
            tjd,
            boresight_radec_rad,
            antenna_positions_xyz,
            longlatalt_rad,
            dut1=dut1,
            astrom=astrom
        )
        boresightUvw -= boresightUvw[reference_antenna_index:reference_antenna_index+1, :]
        for b, beam_coord in enumerate(beam_radec_rad):
            # These UVWs are centered at the reference antenna,
            # i.e. UVW_irefant = [0, 0, 0]
            beamUvw = compute_uvw_from_xyz( # [Antenna, UVW]
                tjd,
                beam_coord,
                antenna_positions_xyz,
                longlatalt_rad,
                dut1=dut1,
                astrom=astrom
            )
            beamUvw -= beamUvw[reference_antenna_index:reference_antenna_index+1, :]

            delays_ns[t, b, :] = (beamUvw[:,2] - boresightUvw[:,2]) * (1e9 / 299792458.0)

    return delays_ns


def phasors_from_delays(
    delays_ns: numpy.ndarray, # [Time, Beam, Antenna]
    frequencies: numpy.ndarray, # [channel-frequencies] Hz
    calibration_coefficients: numpy.ndarray, # [Frequency-channel, Polarization, Antenna]
):
    """
    Return
    ------
        phasors (B, A, F, T, P)
    """

    assert frequencies.shape[0] % calibration_coefficients.shape[0] == 0, f"Calibration Coefficients' Frequency axis is not a factor of frequencies: {calibration_coefficients.shape[0]} vs {frequencies.shape[0]}."

    phasorDims = (
        delays_ns.shape[1],
        delays_ns.shape[2],
        frequencies.shape[0],
        delays_ns.shape[0],
        calibration_coefficients.shape[1]
    )
    calibrationCoeffFreqRatio = frequencies.shape[0] // calibration_coefficients.shape[0]
    calibration_coefficients = numpy.repeat(calibration_coefficients, calibrationCoeffFreqRatio, axis=0) # repeat frequencies

    phasors = numpy.zeros(phasorDims, dtype=numpy.complex128)

    for t in range(delays_ns.shape[0]):
        for b in range(delays_ns.shape[1]):
            for a, delay_ns in enumerate(delays_ns[t, b, :]):
                phasor = numpy.exp(-1.0j*2.0*numpy.pi*delay_ns*1e-9*frequencies)
                phasors[b, a, :, t, :] = numpy.reshape(numpy.repeat(phasor, 2), (len(phasor), 2)) * calibration_coefficients[:, :, a]
    return phasors