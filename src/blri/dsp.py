import numpy

cupy_enabled = False
compy = numpy
correlate_kernel = None


def compute_with_numpy():
    global compy, cupy_enabled
    compy = numpy
    cupy_enabled = False


def compute_with_cupy():
    global compy, cupy_enabled, correlate_kernel
    import cupy
    compy = cupy
    cupy_enabled = True

    correlate_kernel = compy.RawKernel(r'''
    #include <cupy/complex.cuh>

    extern "C" __global__
    void correlate_kernel(
        const int length_FT, const int length_A,
        const complex<float>* x, const complex<float>* x_conj,
        complex<double>* y
    ) {
        int ft_idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (ft_idx >= length_FT) {
            return;
        }

        // auto baseline index
        int antidx_0 = blockIdx.y;
        int antidx_1 = blockIdx.y;

        if (blockIdx.y >= length_A) {
            // cross baseline index
            antidx_0 = ((2*length_A)-1);
            antidx_0 *= antidx_0;
            antidx_0 += (8*length_A) - (8*blockIdx.y);
            double antidx_f0 = sqrt((double)antidx_0);
            antidx_f0 += (-2*length_A)+1;
            antidx_0 = int(antidx_f0/-2.0);

            antidx_1 = blockIdx.y - length_A;
            antidx_1 -= (antidx_0)*(antidx_0+1)/2;
            antidx_1 -= (length_A-1-antidx_0)*antidx_0;
            antidx_1 -= -1-antidx_0;
        }

        y[(blockIdx.y*length_FT + ft_idx)*4 + 0] = x[(antidx_0*length_FT + ft_idx)*2+0] * x_conj[(antidx_1*length_FT + ft_idx)*2+0];
        y[(blockIdx.y*length_FT + ft_idx)*4 + 1] = x[(antidx_0*length_FT + ft_idx)*2+0] * x_conj[(antidx_1*length_FT + ft_idx)*2+1];
        y[(blockIdx.y*length_FT + ft_idx)*4 + 2] = x[(antidx_0*length_FT + ft_idx)*2+1] * x_conj[(antidx_1*length_FT + ft_idx)*2+0];
        y[(blockIdx.y*length_FT + ft_idx)*4 + 3] = x[(antidx_0*length_FT + ft_idx)*2+1] * x_conj[(antidx_1*length_FT + ft_idx)*2+1];
    }
    ''', "correlate_kernel")


def upchannelise(
    datablock: compy.ndarray,
    rate: int
) -> compy.ndarray:
    """
    Args:
        datablock (compy.ndarray):
            the data to be upchannelized in order [Antenna, Frequency, Time, Polarization]
        rate (int):
            the FFT length

    Returns
    -------
    compy.ndarray: [Antenna, Frequency*rate, Time//rate, Polarization]
    """
    global compy

    if rate == 1:
        return datablock

    A, F, T, P = datablock.shape
    assert T % rate == 0, f"Rate {rate} is not a factor of time {T}."
    datablock = datablock.reshape((A, F, T//rate, rate, P))
    datablock = compy.fft.fftshift(compy.fft.fft(
            datablock,
            axis=3
        ),
        axes=3
    )
    return datablock.transpose((0, 1, 3, 2, 4)).reshape((A, F*rate, T//rate, P))


def integrate(
    datablock: compy.ndarray,
    keepdims: bool = False
) -> compy.ndarray:
    """
    Integration of all time samples.

    Args:
        datablock (compy.ndarray):
            the data to be integrated in order [Antenna, Frequency, Time, Polarization]
        keepdims (bool): False
            whether or not the collapsed Time dimension of length 1
            should remain in the shape of the returned data
    """
    return datablock.sum(axis=2, keepdims=keepdims)


def _correlate_antenna_data(
    ant1_data: compy.ndarray,  # [Frequency, Time, Polarization]
    ant2_data: compy.ndarray,  # [Frequency, Time, Polarization]
):
    """
    Produces the correlations with typical polarisation permutation.
    Does not conjugate `ant2_data`.

    Args:
        ant1_data (compy.ndarray):
            data in order [Frequency, Time, Polarization]
        ant2_data (compy.ndarray):
            data in order [Frequency, Time, Polarization]

    Returns:
        compy.ndarray: [Frequency, Time, Polarization*Polarization]
    """
    global compy

    assert ant1_data.shape == ant2_data.shape, "Antenna data must have the same shape"

    P = ant1_data.shape[-1]
    return compy.transpose(
        compy.asarray(
            [
                ant1_data[:, :, p1]*ant2_data[:, :, p2]
                for p1 in range(P) for p2 in range(P)
            ]
        ),
        # shift pol-product dimension to fastest
        (1, 2, 0)
    )


def correlate(
    datablock: compy.ndarray,  # [Antenna, Frequency, Time, Polarization]
):
    """
    Produces the correlations with typical polarisation permutation,
    across all baselines.

    Args:
        datablock (compy.ndarray):
            in order [Antenna, Frequency, Time, Polarization]

    Returns:
        compy.ndarray:
            the auto-baselines come first, followed by the cross-baselines,
            in order [Antenna-Baseline, Frequency, Time, Polarization*Polarization]
    """
    global compy, cupy_enabled, correlate_kernel

    A, F, T, P = datablock.shape
    assert P == 2, f"Expecting 2 polarisations, not {P}."
    assert A > 1, "Expecting more than 1 antenna"

    corr = compy.zeros(
        (
            A*(A+1)//2,
            F,
            T,
            P*P
        ),
        dtype='D'
    )

    if cupy_enabled and not isinstance(datablock, numpy.ndarray):
        assert datablock.dtype == numpy.complex64
        FT = F * T
        threads=(512,)
        blocks=((FT+511)//512, A*(A+1)//2)
        correlate_kernel(blocks, threads, (FT, A, datablock, compy.conj(datablock), corr))
        return corr

    datablock_conj = compy.conjugate(datablock)
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
