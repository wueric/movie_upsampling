import numpy as np
import torch

from typing import Optional, Tuple

from . import upsampling_cpp_lib
from . import torch_sparse_upsample_cuda
from . import jitter_cuda


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


def movie_sparse_upsample_transpose_cuda(movie_frames: torch.Tensor,
                                         frame_selection: torch.Tensor,
                                         frame_weights: torch.Tensor) -> torch.Tensor:
    return torch_sparse_upsample_cuda.movie_sparse_upsample_transpose_cuda(movie_frames, frame_selection, frame_weights)


def flat_sparse_upsample_cuda(movie_frames: torch.Tensor,
                              frame_selection: torch.Tensor,
                              frame_weights: torch.Tensor) -> torch.Tensor:
    return torch_sparse_upsample_cuda.flat_sparse_upsample_cuda(movie_frames, frame_selection, frame_weights)


def flat_sparse_upsample_transpose_cuda(movie_frames: torch.Tensor,
                                        frame_selection: torch.Tensor,
                                        frame_weights: torch.Tensor) -> torch.Tensor:
    return torch_sparse_upsample_cuda.flat_sparse_upsample_transpose_cuda(movie_frames, frame_selection, frame_weights)


def weasel_add(a: torch.Tensor,
               b: torch.Tensor) -> torch.Tensor:
    return torch_sparse_upsample_cuda.test_dumb_add_cuda(a, b)


class JitterFrame(torch.autograd.Function):
    '''
    Wrapper for autograd function that performs forward and backward
        passes for jittering frames
    '''

    @staticmethod
    def forward(ctx,
                input_frames: torch.Tensor,
                coordinates: torch.Tensor) -> torch.Tensor:
        '''

        :param ctx:
        :param batched_frames: shape (batch, height, width)
        :param coordinates: shape (batch, n_jittered_frames, 2), torch.LongTensor dtype
        :return: shape (batch, n_jittered_frames, height, width)
        '''

        ctx.save_for_backward(input_frames, coordinates)
        return jitter_cuda.jitter_movie_forward(input_frames, coordinates)

    @staticmethod
    def backward(ctx,
                 d_loss_d_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:


        input_frames, coordinates = ctx.saved_tensors
        grad_coordinates = None

        grad_image = jitter_cuda.jitter_movie_backward(d_loss_d_output, coordinates)

        return grad_image, grad_coordinates
