import numpy as np

from . import upsampling_cpp_lib

def upsample_transpose_movie(movie_frames: np.ndarray,
                             movie_cutoff_times: np.ndarray,
                             spike_bin_cutoff_times: np.ndarray):
    '''
    Upsamples the movie and transposes the time axis from first to last

    :param movie_frames: shape (n_frames, height, width)
    :param movie_cutoff_times: shape (n_frames + 1, )
    :param spike_bin_cutoff_times: shape (n_bins + 1, )
    :return: shape (height, width, n_bins)
    '''

    return movie_upsampling_cpplib.temporal_upsample_transpose_movie(
        movie_frames, movie_cutoff_times, spike_bin_cutoff_times
    )