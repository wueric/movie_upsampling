import numpy as np
import torch

from typing import Optional, Tuple

from . import upsampling_cpp_lib
from . import torch_sparse_upsample_cuda
from . import jitter_cuda
from . import diff_upsample


def compute_interval_overlaps(movie_cutoff_times: np.ndarray,
                              spike_bin_cutoff_times: np.ndarray):
    '''
    Assumes spike_bin_cutoff_times[0] >= movie_cutoff_times[0] and
        spike_bin_cutoff_times[-1] <= spike_bin_cutoff_times[-1]

    This is because it does not make sense to include empty frames
        in the upsample, since empty frames are incorrect and mean
        that we did not get enough frames from the data

    :param movie_cutoff_times: shape (n_frames + 1, )
    :param spike_bin_cutoff_times: shape (n_bins + 1, )
    :return: shape (height, width, n_bins)
    '''

    assert movie_cutoff_times[0] <= spike_bin_cutoff_times[0]
    assert spike_bin_cutoff_times[-1] <= movie_cutoff_times[-1]

    return upsampling_cpp_lib._compute_interval_overlaps(movie_cutoff_times, spike_bin_cutoff_times)


def batch_compute_interval_overlaps(batched_movie_cutoffs: np.ndarray,
                                    batched_spike_cutoffs: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Assumes spike_bin_cutoff_times[i, 0] >= movie_cutoff_times[i, 0] and
        spike_bin_cutoff_times[i, -1] <= spike_bin_cutoff_times[i, -1]

    This is because it does not make sense to include empty frames
        in the upsample, since empty frames are incorrect and mean
        that we did not get enough frames from the data

    :param batched_movie_cutoffs: shape (batch, n_frame_cutoffs = n_frames + 1)
    :param batched_spike_cutoffs: shape (batch, n_bin_cutoffs = n_bins + 1)
    :return:
    '''
    assert np.all(batched_movie_cutoffs[:, 0] <= batched_spike_cutoffs[:, 0])
    assert np.all(batched_spike_cutoffs[:, -1] <= batched_movie_cutoffs[:, -1])

    return upsampling_cpp_lib._batch_compute_interval_overlaps(batched_movie_cutoffs, batched_spike_cutoffs)


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


class BeamJitterFrame(torch.autograd.Function):
    '''
    Not meant to be differentiable, backward() returns all None
    Wrapper for autograd function that performs forward
        passes for jittering frames for beam search
    '''

    @staticmethod
    def forward(ctx,
                input_frames: torch.Tensor,
                beam_coordinates: torch.Tensor) -> torch.Tensor:
        '''

        :param ctx:
        :param input_frames: shape (beam, height, widht)
        :param beam_coordinates: shape (beam, beam_grid, n_jitter_frames, 2)
        :return:
        '''

        return jitter_cuda.beam_jitter_repeat_frames_forward(
            input_frames,
            beam_coordinates
        )

    @staticmethod
    def backward(ctx, deriv: torch.Tensor):
        return None, None


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
        :param batched_frames: shape (batch, height, width) batched frames to upsample
        :param coordinates: shape (batch, n_jittered_frames, 2), torch.LongTensor dtype
        :return: shape (batch, n_jittered_frames, height, width)
        '''

        ctx.save_for_backward(coordinates)
        return jitter_cuda.jitter_movie_forward(input_frames, coordinates)

    @staticmethod
    def backward(ctx,
                 d_loss_d_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        coordinates, = ctx.saved_tensors
        grad_coordinates = None
        grad_image = jitter_cuda.jitter_movie_backward(d_loss_d_output, coordinates)

        return grad_image, grad_coordinates


class EMJitterFrame(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                batched_input_frames: torch.Tensor,
                batched_grid_coordinates: torch.Tensor) -> torch.Tensor:
        '''

        :param ctx:
        :param batched_input_frames: shape (batch, height, width)
        :param batched_grid_coordinates: shape (batch, n_grid, n_jittered_frames, 2)
        :return: shape (batch, n_grid, n_jittered_frames, height, width)
        '''

        ctx.save_for_backward(batched_grid_coordinates)
        return jitter_cuda.grid_jitter_single_frame_forward(batched_input_frames,
                                                            batched_grid_coordinates)

    @staticmethod
    def backward(ctx, d_output_d_jittered_frame: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        jitter_coords, = ctx.saved_tensors
        backward_frame = jitter_cuda.grid_jitter_single_frame_backward(
            d_output_d_jittered_frame,
            jitter_coords
        )

        return backward_frame, None


class JitterOnlyFrame(torch.autograd.Function):
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
        :param batched_frames: shape (batch, n_jittered_frames, height, width) batched frames to upsample
        :param coordinates: shape (batch, n_jittered_frames, 2), torch.LongTensor dtype
        :return: shape (batch, n_jittered_frames, height, width)
        '''

        ctx.save_for_backward(coordinates)
        return jitter_cuda.jitter_frames_only_forward(input_frames, coordinates)

    @staticmethod
    def backward(ctx,
                 d_loss_d_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        coordinates, = ctx.saved_tensors
        grad_coordinates = None

        grad_image = jitter_cuda.jitter_frames_only_backward(d_loss_d_output, coordinates)

        return grad_image, grad_coordinates


class RepeatFrame(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_frames: torch.Tensor, n_repeats: int) -> torch.Tensor:
        return jitter_cuda.frame_repeat_forward(input_frames, n_repeats)

    @staticmethod
    def backward(ctx, d_loss_d_output: torch.Tensor):
        return jitter_cuda.frame_repeat_backward(d_loss_d_output), None


class TimeUpsampleFlat(torch.autograd.Function):
    '''
    Wrapper for autograd function that performs forwards and backwards passes
        for time-upsampling frames, where the upsample rate may be non-uniform

    This upsamples a flat object; to upsample an actual movie frame
        use TimeUpsampleMovie
    '''

    @staticmethod
    def forward(ctx,
                batch_input_frames: torch.Tensor,
                batch_selection_ix: torch.Tensor,
                batch_sel_weights: torch.Tensor,
                backward_sel: torch.Tensor,
                backward_weights: torch.Tensor) -> torch.Tensor:
        '''

        :param ctx:
        :param batch_input_frames: shape (batch, n_frames_noupsample, n_pix), the flat frame that needs to be
            upsampled. Note that each element in the batch is independently upsampled
        :param batch_selection_ix: shape (batch, n_frames_upsample, 2), type int64_t, precomputed frame selection
            indices into batch_input_frames. Value -1 specifies invalid frame idx
        :param batch_sel_weights: shape (batch, n_frames_upsample, 2), weights for the frames specified in
            batch_selection_ix, last dimension should be nonnegative and sum to 1
        :param backward_sel: shape (batch, n_frames_noupsample, n_max_overlap), type int64_t, precomputed
            frame selection for backward pass. Value -1 specifies invalid frame idx
        :param backward_weights: shape (batch, n_frames_noupsample, n_max_overlap) weights for the frames specified
            in backward_sel, last dimension should be nonnegative
        :return:
        '''

        ctx.save_for_backward(batch_input_frames, batch_selection_ix, batch_sel_weights,
                              backward_sel, backward_weights)
        return diff_upsample.upsample_flat_forward(batch_input_frames,
                                                   batch_selection_ix,
                                                   batch_sel_weights)

    @staticmethod
    def backward(ctx,
                 d_loss_d_upsample) -> Tuple[Optional[torch.Tensor], ...]:
        '''

        :param ctx:
        :param d_loss_d_upsample:
        :return:
        '''

        batch_input, batch_sel, batch_weights, backward_sel, backward_weights = ctx.saved_tensors
        grad_sel, grad_weights, grad_bsel, grad_bweights = None, None, None, None

        grad_noupsample = diff_upsample.upsample_flat_backward(d_loss_d_upsample,
                                                               backward_sel,
                                                               backward_weights)

        return grad_noupsample, grad_sel, grad_weights, grad_bsel, grad_bweights


class ForwardOnlyTimeUpsampleTransposeFlat(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                batch_input_frames: torch.Tensor,
                batch_selection_ix: torch.Tensor,
                batch_sel_weights: torch.Tensor):
        return diff_upsample.upsample_transpose_flat_forward(batch_input_frames,
                                                             batch_selection_ix,
                                                             batch_sel_weights)

    @staticmethod
    def backward(ctx, d_loss_d_upsample: torch.Tensor):
        # designed to be not differentiable
        return None, None, None


class TimeUpsampleTransposeFlat(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                batch_input_frames: torch.Tensor,
                batch_selection_ix: torch.Tensor,
                batch_sel_weights: torch.Tensor,
                backward_sel: torch.Tensor,
                backward_weights: torch.Tensor) -> torch.Tensor:
        '''

        :param ctx:
        :param batch_input_frames: shape (batch, n_frames_noupsample, n_pix), the flat frame that needs to be
            upsampled. Note that each element in the batch is independently upsampled
        :param batch_selection_ix: shape (batch, n_frames_upsample, 2), type int64_t, precomputed frame selection
            indices into batch_input_frames. Value -1 specifies invalid frame idx
        :param batch_sel_weights: shape (batch, n_frames_upsample, 2), weights for the frames specified in
            batch_selection_ix, last dimension should be nonnegative and sum to 1
        :param backward_sel: shape (batch, n_frames_noupsample, n_max_overlap), type int64_t, precomputed
            frame selection for backward pass. Value -1 specifies invalid frame idx
        :param backward_weights: shape (batch, n_frames_noupsample, n_max_overlap) weights for the frames specified
            in backward_sel, last dimension should be nonnegative
        :return: shape (batch, n_pix, n_frames_upsample)
        '''
        batch, n_frames_no_upsample, n_pix = batch_input_frames.shape
        n_frames_upsample = batch_selection_ix.shape[1]

        assert batch_selection_ix.shape == (batch, n_frames_upsample, 2)
        assert batch_sel_weights.shape == (batch, n_frames_upsample, 2)
        assert backward_sel.shape[:2] == (batch, n_frames_no_upsample)
        assert backward_weights.shape[:2] == (batch, n_frames_no_upsample)

        ctx.save_for_backward(backward_sel, backward_weights)

        return diff_upsample.upsample_transpose_flat_forward(batch_input_frames,
                                                             batch_selection_ix,
                                                             batch_sel_weights)

    @staticmethod
    def backward(ctx,
                 d_loss_d_upsample: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        '''

        :param ctx:
        :param d_loss_d_upsample: shape (batch, n_pix, n_frames_upsample)
        :return:
        '''
        backward_sel, backward_weights = ctx.saved_tensors
        grad_sel, grad_weights, grad_bsel, grad_bweights = None, None, None, None

        grad_noupsample = diff_upsample.upsample_transpose_flat_backward(d_loss_d_upsample,
                                                                         backward_sel,
                                                                         backward_weights)
        return grad_noupsample, grad_sel, grad_weights, grad_bsel, grad_bweights


class TimeUpsampleMovie(torch.autograd.Function):
    '''
    Wrapper for autograd function that performs forwards and backwards passes
        for time-upsampling frames, where the upsample rate may be non-uniform

    This one is for 2D movies
    '''

    @staticmethod
    def forward(ctx,
                batch_input_frames: torch.Tensor,
                batch_selection_ix: torch.Tensor,
                batch_sel_weights: torch.Tensor,
                backward_sel: torch.Tensor,
                backward_weights: torch.Tensor) -> torch.Tensor:
        '''

        :param ctx:
        :param batch_input_frames: shape (batch, n_frames_noupsample, height, width), the flat frame that needs to be
            upsampled. Note that each element in the batch is independently upsampled
        :param batch_selection_ix: shape (batch, n_frames_upsample, 2), type int64_t, precomputed frame selection
            indices into batch_input_frames. Value -1 specifies invalid frame idx
        :param batch_sel_weights: shape (batch, n_frames_upsample, 2), weights for the frames specified in
            batch_selection_ix, last dimension should be nonnegative and sum to 1
        :param backward_sel: shape (batch, n_frames_noupsample, n_max_overlap), type int64_t, precomputed
            frame selection for backward pass. Value -1 specifies invalid frame idx
        :param backward_weights: shape (batch, n_frames_noupsample, n_max_overlap) weights for the frames specified
            in backward_sel, last dimension should be nonnegative
        :return:
        '''

        batch, n_frames_noupsample, height, width = batch_input_frames.shape
        flat_input_frames = batch_input_frames.reshape(batch, n_frames_noupsample, height * width)

        ctx.save_for_backward(batch_input_frames, batch_selection_ix, batch_sel_weights,
                              backward_sel, backward_weights)
        return diff_upsample.upsample_flat_forward(flat_input_frames,
                                                   batch_selection_ix,
                                                   batch_sel_weights).reshape(batch, -1, height, width)

    @staticmethod
    def backward(ctx,
                 d_loss_d_upsample) -> Tuple[Optional[torch.Tensor], ...]:
        '''

        :param ctx:
        :param d_loss_d_upsample:
        :return:
        '''

        batch_input, batch_sel, batch_weights, backward_sel, backward_weights = ctx.saved_tensors
        orig_batch, orig_n_frames_noupsample, orig_height, orig_width = batch_input.shape
        grad_sel, grad_weights, grad_bsel, grad_bweights = None, None, None, None

        grad_noupsample_unshape = diff_upsample.upsample_flat_backward(
            d_loss_d_upsample.reshape(orig_batch, -1, orig_height * orig_width),
            backward_sel,
            backward_weights)
        grad_noupsample = grad_noupsample_unshape.reshape(
            orig_batch, orig_n_frames_noupsample, orig_height, orig_width)

        return grad_noupsample, grad_sel, grad_weights, grad_bsel, grad_bweights


class TimeUpsampleTransposeMovie(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                batch_input_frames: torch.Tensor,
                batch_selection_ix: torch.Tensor,
                batch_sel_weights: torch.Tensor,
                backward_sel: torch.Tensor,
                backward_weights: torch.Tensor) -> torch.Tensor:
        '''

        :param ctx:
        :param batch_input_frames: shape (batch, n_frames_noupsample, height, width), the flat frame that needs to be
            upsampled. Note that each element in the batch is independently upsampled
        :param batch_selection_ix: shape (batch, n_frames_upsample, 2), type int64_t, precomputed frame selection
            indices into batch_input_frames. Value -1 specifies invalid frame idx
        :param batch_sel_weights: shape (batch, n_frames_upsample, 2), weights for the frames specified in
            batch_selection_ix, last dimension should be nonnegative and sum to 1
        :param backward_sel: shape (batch, n_frames_noupsample, n_max_overlap), type int64_t, precomputed
            frame selection for backward pass. Value -1 specifies invalid frame idx
        :param backward_weights: shape (batch, n_frames_noupsample, n_max_overlap) weights for the frames specified
            in backward_sel, last dimension should be nonnegative
        :return: shape (batch, height, width, n_frames_upsample)
        '''

        batch, n_frames_noupsample, height, width = batch_input_frames.shape
        flat_input_frames = batch_input_frames.reshape(batch, n_frames_noupsample, height * width)

        ctx.save_for_backward(batch_input_frames, batch_selection_ix, batch_sel_weights,
                              backward_sel, backward_weights)
        return diff_upsample.upsample_transpose_flat_forward(flat_input_frames,
                                                             batch_selection_ix,
                                                             batch_sel_weights).reshape(batch, height, width, -1)

    @staticmethod
    def backward(ctx,
                 d_loss_d_upsample) -> Tuple[Optional[torch.Tensor], ...]:
        '''

        :param ctx:
        :param d_loss_d_upsample: shape (batch, height, width, n_frames_upsample)
        :return: shape (batch, n_frames_noupsample, height, width)
        '''

        batch_input, batch_sel, batch_weights, backward_sel, backward_weights = ctx.saved_tensors
        orig_batch, orig_n_frames_noupsample, orig_height, orig_width = batch_input.shape
        grad_sel, grad_weights, grad_bsel, grad_bweights = None, None, None, None

        grad_noupsample_unshape = diff_upsample.upsample_transpose_flat_backward(
            d_loss_d_upsample.reshape(orig_batch, orig_height * orig_width, -1),
            backward_sel,
            backward_weights)
        grad_noupsample = grad_noupsample_unshape.reshape(
            orig_batch, orig_n_frames_noupsample, orig_height, orig_width)

        return grad_noupsample, grad_sel, grad_weights, grad_bsel, grad_bweights


class SharedClockTimeUpsampleTransposeFlat(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                batch_input_frames: torch.Tensor,
                share_selection_ix: torch.Tensor,
                share_sel_weights: torch.Tensor,
                share_backward_sel: torch.Tensor,
                share_backward_weights: torch.Tensor) -> torch.Tensor:
        '''

        :param ctx:
        :param batch_input_frames: shape (batch, n_frames_noupsample, n_pix)
        :param share_selection_ix: shape (n_frames_upsample, 2), int64
        :param share_sel_weights: shape (n_frames_upsample, 2)
        :param share_backward_sel: shape (n_frames_noupsample, n_max_overlap), int64
        :param share_backward_weights: shape (n_frames_noupsample, n_max_overlap)
        :return: shape (batch, n_pix, n_frames_upsample)
        '''

        batch, n_frames_no_upsample, n_pix = batch_input_frames.shape
        n_frames_upsample = share_selection_ix.shape[0]

        assert share_selection_ix.shape == (n_frames_upsample, 2)
        assert share_sel_weights.shape == (n_frames_upsample, 2)
        assert share_backward_sel.shape[0] == n_frames_no_upsample
        assert share_backward_weights.shape[0] == n_frames_no_upsample

        ctx.save_for_backward(share_backward_sel, share_backward_weights)

        return diff_upsample.shared_clock_time_upsample_transpose_forward(batch_input_frames,
                                                                          share_selection_ix,
                                                                          share_sel_weights)

    @staticmethod
    def backward(ctx,
                 d_loss_d_upsample) -> Tuple[Optional[torch.Tensor], ...]:

        share_backward_sel, share_backward_weights = ctx.saved_tensors
        grad_noupsample = diff_upsample.shared_clock_upsample_transpose_flat_backward(
            d_loss_d_upsample,
            share_backward_sel,
            share_backward_weights
        )

        return grad_noupsample, None, None, None, None

