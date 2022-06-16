import numpy as np
import torch

from . import upsampling_cpp_lib
from . import torch_sparse_upsample_cuda


def compute_interval_overlaps(movie_cutoff_times: np.ndarray,
                              spike_bin_cutoff_times: np.ndarray):
    '''
    Upsamples the movie and transposes the time axis from first to last

    :param movie_cutoff_times: shape (n_frames + 1, )
    :param spike_bin_cutoff_times: shape (n_bins + 1, )
    :return: shape (height, width, n_bins)
    '''

    return upsampling_cpp_lib._compute_interval_overlaps(movie_cutoff_times, spike_bin_cutoff_times)


def movie_sparse_upsample_cuda(movie_frames: torch.Tensor,
                               frame_selection: torch.Tensor,
                               frame_weights: torch.Tensor) -> torch.Tensor:
    return torch_sparse_upsample_cuda.movie_sparse_upsample_cuda(movie_frames, frame_selection, frame_weights)


def weasel_add(a: torch.Tensor,
               b: torch.Tensor) -> torch.Tensor:
    return torch_sparse_upsample_cuda.test_dumb_add_cuda(a, b)
