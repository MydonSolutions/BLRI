import numpy


def upchannelise(
    datablock: numpy.ndarray,
    rate: int
) -> numpy.ndarray:
    """
    Args:
        datablock (numpy.ndarray):
            the data to be upchannelized in order [Antenna, Frequency, Time, Polarization]
        rate (int):
            the FFT length

    Returns
    -------
    numpy.ndarray: [Antenna, Frequency*rate, Time//rate, Polarization]
    """
    if rate == 1:
        return datablock

    A, F, T, P = datablock.shape
    assert T % rate == 0, f"Rate {rate} is not a factor of time {T}."
    datablock = datablock.reshape((A, F, T//rate, rate, P))
    datablock = numpy.fft.fftshift(numpy.fft.fft(
            datablock,
            axis=3
        ),
        axes=3
    )
    return datablock.transpose((0, 1, 3, 2, 4)).reshape((A, F*rate, T//rate, P))


def integrate(
    datablock: numpy.ndarray,
    keepdims: bool = False
) -> numpy.ndarray:
    """
    Integration of all time samples.

    Args:
        datablock (numpy.ndarray):
            the data to be integrated in order [Antenna, Frequency, Time, Polarization]
        keepdims (bool): False
            whether or not the collapsed Time dimension of length 1
            should remain in the shape of the returned data
    """
    return datablock.sum(axis=2, keepdims=keepdims)


def _correlate_antenna_data(
    ant1_data: numpy.ndarray,  # [Frequency, Time, Polarization]
    ant2_data: numpy.ndarray,  # [Frequency, Time, Polarization]
):
    """
    Produces the correlations with typical polarisation permutation.
    Does not conjugate `ant2_data`.

    Args:
        ant1_data (numpy.ndarray):
            data in order [Frequency, Time, Polarization]
        ant2_data (numpy.ndarray):
            data in order [Frequency, Time, Polarization]

    Returns:
        numpy.ndarray: [Frequency, Time, Polarization*Polarization]
    """
    assert ant1_data.shape == ant2_data.shape, "Antenna data must have the same shape"

    P = ant1_data.shape[-1]
    return numpy.transpose(
        [
            ant1_data[:, :, p1]*ant2_data[:, :, p2]
            for p1 in range(P) for p2 in range(P)
        ],
        # shift pol-product dimension to fastest
        (1, 2, 0)
    )


def correlate(
    datablock: numpy.ndarray,  # [Antenna, Frequency, Time, Polarization]
):
    """
    Produces the correlations with typical polarisation permutation,
    across all baselines.

    Args:
        datablock (numpy.ndarray):
            in order [Antenna, Frequency, Time, Polarization]

    Returns:
        numpy.ndarray:
            the auto-baselines come first, followed by the cross-baselines,
            in order [Antenna-Baseline, Frequency, Time, Polarization*Polarization]
    """
    A, F, T, P = datablock.shape
    assert P == 2, f"Expecting 2 polarisations, not {P}."
    assert A > 1, "Expecting more than 1 antenna"

    datablock_conj = numpy.conjugate(datablock)

    corr = numpy.zeros(
        (
            A*(A+1)//2,
            F,
            T,
            P*P
        ),
        dtype='D'
    )
    correlation_index = 0

    # auto correlations first
    for a in range(A):
        corr[correlation_index, :, :, :] = _correlate_antenna_data(
            datablock[a, :, :, :],
            datablock_conj[a, :, :, :]
        )
        correlation_index += 1

    # cross correlations
    for a1 in range(A):
        for a2 in range(a1+1, A):
            corr[correlation_index, :, :, :] = _correlate_antenna_data(
                datablock[a1, :, :, :],
                datablock_conj[a2, :, :, :]
            )
            correlation_index += 1

    return corr
