import numpy as np

from . import upsampling_cpp_lib

def compute_interval_overlaps(movie_cutoff_times: np.ndarray,
                              spike_bin_cutoff_times: np.ndarray):
    '''
    Upsamples the movie and transposes the time axis from first to last

    :param movie_cutoff_times: shape (n_frames + 1, )
    :param spike_bin_cutoff_times: shape (n_bins + 1, )
    :return: shape (height, width, n_bins)
    '''

    return upsampling_cpp_lib.compute_interval_overlaps(
        movie_cutoff_times, spike_bin_cutoff_times
    )
