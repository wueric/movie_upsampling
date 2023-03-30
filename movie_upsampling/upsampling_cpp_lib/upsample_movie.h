//
// Created by Eric Wu on 6/13/22.
//

#ifndef MOVIE_UPSAMPLING_UPSAMPLE_MOVIE_H
#define MOVIE_UPSAMPLING_UPSAMPLE_MOVIE_H

#define INVALID_FRAME -1

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <tuple>
#include <iostream>

#include "NDArrayWrapper.h"

namespace py = pybind11;

template<class T>
using ContigNPArray = py::array_t<T, py::array::c_style | py::array::forcecast>;


template<class T>
void _raw_integer_upsample_movie(
        CNDArrayWrapper::StaticNDArrayWrapper<T, 2> movie_flat_noupsample,
        CNDArrayWrapper::StaticNDArrayWrapper<T, 2> movie_flat_upsampled,
        int64_t upsample_factor) {

    /*
     * Code to do dumb integer scale upsampling (i.e. 2x upsampling, 3x upsampling,
     * 4x upsampling) of the movie frames
     *
     * Obviously this method does not attempt to account for frame rate instabilities
     *
     * Primary use case: temporally upsampling white noise stimuli in a high-performance way
     *
     *
     * movie_flat_noupsample: shape (n_frames, n_pixels)
     * movie_flat_upsampled: shape (upsample_factor * n_frames, n_pixels)
     *
     */

    int64_t n_frames_noupsample = movie_flat_noupsample.shape[0];
    int64_t n_pix_upsample = movie_flat_noupsample.shape[1];

    for (int64_t t = 0; t < n_frames_noupsample; ++t) {
        for (int64_t p = 0; p < n_pix_upsample; ++p) {

            T val_read = movie_flat_noupsample.template valueAt(t, p);

            for (int64_t u = 0; u < upsample_factor; ++u) {
                movie_flat_upsampled.template storeTo(val_read, t * upsample_factor + u, p);
            }
        }
    }
}


template<class T>
void _raw_integer_upsample_transpose_movie(
        CNDArrayWrapper::StaticNDArrayWrapper<T, 2> movie_flat_noupsample,
        CNDArrayWrapper::StaticNDArrayWrapper<T, 2> movie_flat_upsampled,
        int64_t upsample_factor) {

    /*
     * Code to do dumb integer scale upsampling (i.e. 2x upsampling, 3x upsampling,
     * 4x upsampling) of the movie frames
     *
     * Also flips the time axis to the last dimension, so that temporal convolutions can
     * be faster
     *
     * Obviously this method does not attempt to account for frame rate instabilities
     *
     * Primary use case: temporally upsampling white noise stimuli in a high-performance way
     *
     *
     * movie_flat_noupsample: shape (n_frames, n_pixels)
     * movie_flat_upsampled: shape (n_pixels, upsample_factor * n_frames)
     *
     */

    int64_t n_frames_noupsample = movie_flat_noupsample.shape[0];
    int64_t n_pix_upsample = movie_flat_noupsample.shape[1];

    for (int64_t t = 0; t < n_frames_noupsample; ++t) {
        for (int64_t p = 0; p < n_pix_upsample; ++p) {

            T val_read = movie_flat_noupsample.template valueAt(t, p);

            for (int64_t u = 0; u < upsample_factor; ++u) {
                movie_flat_upsampled.template storeTo(val_read, p, t * upsample_factor + u);
            }
        }
    }
}


template<class F>
ContigNPArray<F> _integer_upsample_transpose_movie(
        ContigNPArray<F> movie_flat,
        int64_t upsample_factor) {

    py::buffer_info movie_info = movie_flat.request();
    auto *movie_flat_ptr = static_cast<F *>(movie_info.ptr);
    const int64_t n_frames_orig = movie_info.shape[0];
    const int64_t n_pix = movie_info.shape[1];

    CNDArrayWrapper::StaticNDArrayWrapper<F, 2> orig_movie_wrapper(movie_flat_ptr,
                                                                   std::array<int64_t, 2>({n_frames_orig, n_pix}));

    auto frame_weight_info = py::buffer_info(
            nullptr,
            sizeof(F),
            py::format_descriptor<F>::value,
            2, /* How many dimensions */
            {static_cast<py::ssize_t>(n_pix), static_cast<py::ssize_t>(n_frames_orig * upsample_factor)}, /* shape */
            {static_cast<py::ssize_t>(n_frames_orig * upsample_factor * sizeof(F)),
             static_cast<py::ssize_t>(sizeof(F))} /* stride */
    );
    ContigNPArray<F> frame_weights = ContigNPArray<F>(frame_weight_info);

    CNDArrayWrapper::StaticNDArrayWrapper<F, 2> transpose_upsample_movie_wrapper(
            static_cast<F *>(frame_weights.request().ptr),
            std::array<int64_t, 2>({n_pix, n_frames_orig * upsample_factor})
    );

    _raw_integer_upsample_transpose_movie(orig_movie_wrapper,
                                          transpose_upsample_movie_wrapper,
                                          upsample_factor);

    return frame_weights;
}

template<class F>
ContigNPArray<F> _integer_upsample_movie(
        ContigNPArray<F> movie_flat,
        int64_t upsample_factor) {

    py::buffer_info movie_info = movie_flat.request();
    auto *movie_flat_ptr = static_cast<F *>(movie_info.ptr);
    const int64_t n_frames_orig = movie_info.shape[0];
    const int64_t n_pix = movie_info.shape[1];

    CNDArrayWrapper::StaticNDArrayWrapper<F, 2> orig_movie_wrapper(movie_flat_ptr,
                                                                   std::array<int64_t, 2>({n_frames_orig, n_pix}));

    auto frame_weight_info = py::buffer_info(
            nullptr,
            sizeof(F),
            py::format_descriptor<F>::value,
            2, /* How many dimensions */
            {static_cast<py::ssize_t>(n_frames_orig * upsample_factor), static_cast<py::ssize_t>(n_pix)}, /* shape */
            {static_cast<py::ssize_t>(n_pix * sizeof(F)), static_cast<py::ssize_t>(sizeof(F))} /* stride */
    );
    ContigNPArray<F> frame_weights = ContigNPArray<F>(frame_weight_info);

    CNDArrayWrapper::StaticNDArrayWrapper<F, 2> transpose_upsample_movie_wrapper(
            static_cast<F *>(frame_weights.request().ptr),
            std::array<int64_t, 2>({n_frames_orig * upsample_factor, n_pix})
    );

    _raw_integer_upsample_movie(orig_movie_wrapper,
                                transpose_upsample_movie_wrapper,
                                upsample_factor);

    return frame_weights;
}


template<class T>
int64_t _raw_compute_interval_overlaps(CNDArrayWrapper::StaticNDArrayWrapper<T, 1> movie_bin_cutoffs,
                                       CNDArrayWrapper::StaticNDArrayWrapper<T, 1> spike_bin_cutoffs,
                                       CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 2> output_overlaps,
                                       CNDArrayWrapper::StaticNDArrayWrapper<T, 2> frame_weights) {
    /*
     * To support batching where there may be spike bins that do not overlap with frames,
     *      we are modifying this function
     *
     * There are now four distinct cases for a given spike bin
     *
     * Overlap cases:
     * (1)  the spike bin occurs completely within a frame, i.e.
     *          no frame transition occurs within the spike bin
     * (2)  a frame transition occurs within the spike bin
     *
     *
     * Non-overlap cases:
     * (3) the spike bin occurs entirely before the first frame.
     *     In this case all frame weights should be zero, and the
     *     corresponding value in output_overlaps can be anything
     *     that corresponds to a valid frame, since it will be
     *     multiplied by zero
     *
     * (4) the spike bin occurs entirely after the last frame
     *     In this case all frame weights should be zero, and the
     *     corresponding value in output_overlaps can be anything
     *     that corresponds to a valid frame, since it will be
     *     multiplied by zero
     *
     *
     * @param movie_bin_cutoffs: shape (n_frames + 1, )
     *      movie frame transition times, must be in increasing order
     * @param spike_bin_cutoffs: shape (n_spike_bins + 1, )
     *      spike bin edge times, must be in increasing order
     * @param output_overlaps: shape (n_spike_bins, 2), int64-valued
     *      index of the frames that the corresponding spike bin
     *      overlaps with. Since the spike bins are assumed to be always
     *      smaller than the frame bins, each spike bin can overlap with
     *      at most 2 frames
     *
     * @param frame_weights: shape (n_spike_bins, 2),
     *      multiplicative weight corresponding to the degree of overlap
     *      between the spike bin and the frames specified in output_overlaps
     */

    int64_t n_spike_bins = spike_bin_cutoffs.shape[0] - 1;
    int64_t n_frames = movie_bin_cutoffs.shape[0] - 1;

    int64_t max_overlapping_bins = 0;
    int64_t curr_frame_n_bins_overlapped = 0;

    int64_t us_idx = 0;
    while (us_idx < n_spike_bins &&
           spike_bin_cutoffs.valueAt(us_idx + 1) < movie_bin_cutoffs.valueAt(0)) {
        // in this loop, the spike bins occur completely before the first frame
        // this loop takes care of case (3) from the documentation
        output_overlaps.storeTo(0, us_idx, 0);
        output_overlaps.storeTo(0, us_idx, 1); // these value should never be used
        // simply a placeholder that corresponds to a valid memory address
        // so that we can avoid branching in the CUDA code

        frame_weights.storeTo(0.0, us_idx, 0);
        frame_weights.storeTo(0.0, us_idx, 1);

        ++us_idx;
    }

    int64_t frame_idx = 0;
    while (us_idx < n_spike_bins &&
           spike_bin_cutoffs.valueAt(us_idx) < movie_bin_cutoffs.valueAt(n_frames)) {

        // in this loop, the spike bins overlap with at least one frame
        // this loop takes care of cases (1) and (2) from the documentation
        T low = spike_bin_cutoffs.valueAt(us_idx);
        T high = spike_bin_cutoffs.valueAt(us_idx + 1);
        T bin_width = high - low;

        /* Determine which movie frames this interval overlaps with
         * Because this function is guaranteed to be upsampling movies
         * this interval is guaranteed to overlap with either one or two
         * movie frames
         *
         * We also count the maximum number of bins that are overlapped by any single
         * frame so that we can compute the overlap selection and weights for the backward pass
         */
        int64_t frame_low = frame_idx;
        while ((frame_low < (n_frames - 1)) && (movie_bin_cutoffs.valueAt(frame_low + 1) < low)) ++frame_low;

        int64_t frame_high = frame_low;
        T curr_frame_start = movie_bin_cutoffs.valueAt(frame_high);
        T curr_frame_end = movie_bin_cutoffs.valueAt(frame_high + 1);

        if (curr_frame_start <= low && curr_frame_end >= high) {
            // in this case, the bin occurs entirely within a frame,
            // i.e. no frame transition occurs
            output_overlaps.storeTo(frame_high, us_idx, 0);
            frame_weights.storeTo(1.0, us_idx, 0);

            output_overlaps.storeTo(frame_high, us_idx, 1); // this value should never be used
            // simply a placeholder that corresponds to a valid memory address
            // so that we can avoid branching in the CUDA code
            frame_weights.storeTo(0.0, us_idx, 1);

            frame_idx = frame_high;

            ++curr_frame_n_bins_overlapped;
            max_overlapping_bins = std::max(max_overlapping_bins, curr_frame_n_bins_overlapped);
        } else {
            // in this case, the bin occurs during a frame transition
            T interval_overlap = (std::min(curr_frame_end, high) - std::max(curr_frame_start, low)) / bin_width;
            output_overlaps.storeTo(frame_high, us_idx, 0);
            frame_weights.storeTo(interval_overlap, us_idx, 0);

            ++frame_high;
            output_overlaps.storeTo(frame_high, us_idx, 1);
            frame_weights.storeTo(1.0 - interval_overlap, us_idx, 1);

            frame_idx = frame_high;

            max_overlapping_bins = std::max(max_overlapping_bins, 1 + curr_frame_n_bins_overlapped);
            curr_frame_n_bins_overlapped = 1;
        }

        ++us_idx
    }

    while (us_idx < n_spike_bins) {
        // in this loop, the spike bins occur completely after the last frame
        // this loop takes care of case (4) from the documentation
        output_overlaps.storeTo(n_frames - 1, us_idx, 0);
        output_overlaps.storeTo(n_frames - 1, us_idx, 1); // these value should never be used
        // simply a placeholder that corresponds to a valid memory address
        // so that we can avoid branching in the CUDA code

        frame_weights.storeTo(0.0, us_idx, 0);
        frame_weights.storeTo(0.0, us_idx, 1);

        ++us_idx;
    }

    return max_overlapping_bins;
}


template<class T>
void _raw_compute_backward_interval_overlaps(
        CNDArrayWrapper::StaticNDArrayWrapper<T, 1> movie_bin_cutoffs,
        CNDArrayWrapper::StaticNDArrayWrapper<T, 1> spike_bin_cutoffs,
        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 2> backward_overlaps,
        CNDArrayWrapper::StaticNDArrayWrapper<T, 2> backward_weights) {
    /*
     * @param movie_bin_cutoffs: shape (n_movie_frame_cutoffs = n_movie_frames + 1, )
     * @param spike_bin_cutoffs: shape (n_spike_bin_cutoffs = n_spike_bins + 1, )
     * @param backward_overlaps: shape (n_movie_frames, n_bins_max_overlap)
     * @param backward_weights: shape (n_movie_frames, n_bins_max_overlap)
     */

    int64_t n_spike_bins = spike_bin_cutoffs.shape[0] - 1;
    int64_t n_frames = movie_bin_cutoffs.shape[0] - 1;
    int64_t max_n_overlaps = backward_overlaps.shape[1];

    int64_t sbin = 0;
    for (int64_t mv = 0; mv < n_frames; ++mv) {
        T frame_start = movie_bin_cutoffs.template valueAt(mv);
        T frame_end = movie_bin_cutoffs.template valueAt(mv + 1);

        // find the first spike bin for which this overlaps
        while ((sbin < n_spike_bins - 1) && spike_bin_cutoffs.template valueAt(sbin + 1) < frame_start) ++sbin;

        int64_t overlap_count = 0;
        while (overlap_count < max_n_overlaps && sbin < n_spike_bins &&
               spike_bin_cutoffs.template valueAt(sbin) <= frame_end) {

            T low = spike_bin_cutoffs.template valueAt(sbin);
            T high = spike_bin_cutoffs.template valueAt(sbin + 1);
            T bin_width = high - low;
            T interval_overlap = (std::min(frame_end, high) - std::max(frame_start, low)) / bin_width;

            backward_overlaps.template storeTo(sbin, mv, overlap_count);
            backward_weights.template storeTo(interval_overlap, mv, overlap_count);
            ++overlap_count;

            ++sbin;
        }

        --sbin;
        // fill the remaining entries (for unused bin slots) with values
        // that don't affect the output, rather than INVALID_FRAME so that
        // we can get rid of a branch in the GPU code
        for (; overlap_count < max_n_overlaps; ++overlap_count) {
            backward_overlaps.template storeTo(sbin, mv, overlap_count);
            backward_weights.template storeTo(0.0, mv, overlap_count);
        }
    }
}


template<class F>
std::tuple <ContigNPArray<int64_t>, ContigNPArray<F>> _compute_interval_overlaps(
        ContigNPArray<F> movie_bin_cutoffs,
        ContigNPArray<F> spike_bin_cutoffs) {

    py::buffer_info movie_bin_info = movie_bin_cutoffs.request();
    auto *movie_bin_ptr = static_cast<F *>(movie_bin_info.ptr);
    const int64_t n_frame_cutoffs = movie_bin_info.shape[0];

    CNDArrayWrapper::StaticNDArrayWrapper<F, 1> movie_bin_wrapper(
            movie_bin_ptr,
            {n_frame_cutoffs});

    py::buffer_info spike_bin_info = spike_bin_cutoffs.request();
    auto *spike_bin_ptr = static_cast<F *>(spike_bin_info.ptr);
    const int64_t n_bin_cutoffs = spike_bin_info.shape[0];
    const int64_t n_bins = n_bin_cutoffs - 1;

    CNDArrayWrapper::StaticNDArrayWrapper<F, 1> spike_bin_wrapper(
            spike_bin_ptr,
            {n_bin_cutoffs});

    auto frame_weight_info = py::buffer_info(
            nullptr,
            sizeof(F),
            py::format_descriptor<F>::value,
            2, /* How many dimensions */
            {static_cast<py::ssize_t>(n_bins), static_cast<py::ssize_t>(2)}, /* shape */
            {static_cast<py::ssize_t>(2 * sizeof(F)), static_cast<py::ssize_t>(sizeof(F))} /* stride */
    );
    ContigNPArray<F> frame_weights = ContigNPArray<F>(frame_weight_info);
    CNDArrayWrapper::StaticNDArrayWrapper<F, 2> frame_weight_wrapper(
            static_cast<F *>(frame_weights.request().ptr),
            std::array<int64_t, 2>({n_bins, 2}));

    auto frame_idx_info = py::buffer_info(
            nullptr,
            sizeof(int64_t),
            py::format_descriptor<int64_t>::value,
            2, /* How many dimensions */
            {static_cast<py::ssize_t>(n_bins), static_cast<py::ssize_t>(2)}, /* shape */
            {static_cast<py::ssize_t>(2 * sizeof(int64_t)), static_cast<py::ssize_t>(sizeof(int64_t))} /* stride */
    );
    ContigNPArray<int64_t> frame_ix = ContigNPArray<int64_t>(frame_idx_info);
    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 2> frame_ix_wrapper(static_cast<int64_t *>(frame_ix.request().ptr),
                                                                       std::array<int64_t, 2>({n_bins, 2}));

    _raw_compute_interval_overlaps<F>(movie_bin_wrapper,
                                      spike_bin_wrapper,
                                      frame_ix_wrapper,
                                      frame_weight_wrapper);

    return std::make_pair(frame_ix, frame_weights);

}


template<class F>
std::tuple <ContigNPArray<int64_t>, ContigNPArray<F>, ContigNPArray<int64_t>, ContigNPArray<F>>
_batch_compute_interval_overlaps(ContigNPArray<F> batched_movie_bin_cutoffs,
                                 ContigNPArray<F> batched_spike_bin_cutoffs) {

    py::buffer_info batched_movie_bin_info = batched_movie_bin_cutoffs.request();
    auto *batched_movie_bin_ptr = static_cast<F *>(batched_movie_bin_info.ptr);
    const int64_t batch = batched_movie_bin_info.shape[0];
    const int64_t n_frame_cutoffs = batched_movie_bin_info.shape[1];
    const int64_t n_frames = n_frame_cutoffs - 1;

    CNDArrayWrapper::StaticNDArrayWrapper<F, 2> movie_bin_wrapper(batched_movie_bin_ptr,
                                                                  std::array<int64_t, 2>({batch, n_frame_cutoffs}));


    py::buffer_info batched_spike_bin_info = batched_spike_bin_cutoffs.request();
    auto *batched_spike_bin_ptr = static_cast<F *>(batched_spike_bin_info.ptr);
    const int64_t n_bin_cutoffs = batched_spike_bin_info.shape[1];
    const int64_t n_bins = n_bin_cutoffs - 1;

    CNDArrayWrapper::StaticNDArrayWrapper<F, 2> spike_bin_wrapper(batched_spike_bin_ptr,
                                                                  std::array<int64_t, 2>({batch, n_bin_cutoffs}));

    auto frame_weight_info = py::buffer_info(
            nullptr,
            sizeof(F),
            py::format_descriptor<F>::value,
            3, /* How many dimensions */
            {batch, static_cast<py::ssize_t>(n_bins), static_cast<py::ssize_t>(2)}, /* shape */
            {static_cast<py::ssize_t> (2 * sizeof(F) * n_bins), static_cast<py::ssize_t>(2 * sizeof(F)),
             static_cast<py::ssize_t>(sizeof(F))} /* stride */
    );
    ContigNPArray<float> frame_weights = ContigNPArray<F>(frame_weight_info);
    CNDArrayWrapper::StaticNDArrayWrapper<F, 3> frame_weight_wrapper(static_cast<F *>(frame_weights.request().ptr),
                                                                     std::array<int64_t, 3>({batch, n_bins, 2}));

    auto frame_idx_info = py::buffer_info(
            nullptr,
            sizeof(int64_t),
            py::format_descriptor<int64_t>::value,
            3, /* How many dimensions */
            {batch, static_cast<py::ssize_t>(n_bins), static_cast<py::ssize_t>(2)}, /* shape */
            {static_cast<py::ssize_t> (2 * sizeof(int64_t) * n_bins), static_cast<py::ssize_t>(2 * sizeof(int64_t)),
             static_cast<py::ssize_t>(sizeof(int64_t))} /* stride */
    );

    ContigNPArray<int64_t> frame_ix = ContigNPArray<int64_t>(frame_idx_info);
    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 3> frame_ix_wrapper(static_cast<int64_t *>(frame_ix.request().ptr),
                                                                       std::array<int64_t, 3>({batch, n_bins, 2}));

    int64_t max_bins_for_frame = 0;
    for (int64_t b = 0; b < batch; ++b) {
        int64_t max_bins_for_trial = _raw_compute_interval_overlaps<F>(
                movie_bin_wrapper.template slice<1>(CNDArrayWrapper::makeIdxSlice(b),
                                                    CNDArrayWrapper::makeAllSlice()),
                spike_bin_wrapper.template slice<1>(CNDArrayWrapper::makeIdxSlice(b),
                                                    CNDArrayWrapper::makeAllSlice()),
                frame_ix_wrapper.template slice<2>(CNDArrayWrapper::makeIdxSlice(b),
                                                   CNDArrayWrapper::makeAllSlice(),
                                                   CNDArrayWrapper::makeAllSlice()),
                frame_weight_wrapper.template slice<2>(CNDArrayWrapper::makeIdxSlice(b),
                                                       CNDArrayWrapper::makeAllSlice(),
                                                       CNDArrayWrapper::makeAllSlice())
        );

        max_bins_for_frame = std::max(max_bins_for_frame, max_bins_for_trial);
    }

    auto backward_weight_info = py::buffer_info(
            nullptr,
            sizeof(F),
            py::format_descriptor<F>::value,
            3, /* How many dimensions */
            {batch, static_cast<py::ssize_t>(n_frames), static_cast<py::ssize_t>(max_bins_for_frame)}, /* shape */
            {static_cast<py::ssize_t> (sizeof(F) * max_bins_for_frame * n_frames),
             static_cast<py::ssize_t>(max_bins_for_frame * sizeof(F)),
             static_cast<py::ssize_t>(sizeof(F))} /* stride */
    );

    ContigNPArray<F> backward_weights = ContigNPArray<F>(backward_weight_info);
    CNDArrayWrapper::StaticNDArrayWrapper<F, 3> backward_weight_wrapper(
            static_cast<F *>(backward_weights.request().ptr),
            std::array<int64_t, 3>({batch, n_frames, max_bins_for_frame}));

    auto backward_sel_info = py::buffer_info(
            nullptr,
            sizeof(int64_t),
            py::format_descriptor<int64_t>::value,
            3, /* How many dimensions */
            {batch, static_cast<py::ssize_t>(n_frames), static_cast<py::ssize_t>(max_bins_for_frame)}, /* shape */
            {static_cast<py::ssize_t> (sizeof(int64_t) * max_bins_for_frame * n_frames),
             static_cast<py::ssize_t>(max_bins_for_frame * sizeof(int64_t)),
             static_cast<py::ssize_t>(sizeof(int64_t))} /* stride */
    );
    ContigNPArray<int64_t> backward_sel = ContigNPArray<int64_t>(backward_sel_info);
    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 3> backward_sel_wrapper(
            static_cast<int64_t *>(backward_sel.request().ptr),
            std::array<int64_t, 3>({batch, n_frames, max_bins_for_frame}));

    for (int64_t b = 0; b < backward_sel_wrapper.shape[0]; ++b) {
        for (int64_t i = 0; i < backward_sel_wrapper.shape[1]; ++i) {
            for (int64_t j = 0; j < backward_sel_wrapper.shape[2]; ++j) {
                backward_sel_wrapper.template storeTo(INVALID_FRAME, b, i, j);
            }
        }
    }

    for (int64_t b = 0; b < batch; ++b) {
        _raw_compute_backward_interval_overlaps<F>(
                movie_bin_wrapper.template slice<1>(CNDArrayWrapper::makeIdxSlice(b),
                                                    CNDArrayWrapper::makeAllSlice()),
                spike_bin_wrapper.template slice<1>(CNDArrayWrapper::makeIdxSlice(b),
                                                    CNDArrayWrapper::makeAllSlice()),
                backward_sel_wrapper.template slice<2>(CNDArrayWrapper::makeIdxSlice(b),
                                                       CNDArrayWrapper::makeAllSlice(),
                                                       CNDArrayWrapper::makeAllSlice()),
                backward_weight_wrapper.template slice<2>(CNDArrayWrapper::makeIdxSlice(b),
                                                          CNDArrayWrapper::makeAllSlice(),
                                                          CNDArrayWrapper::makeAllSlice()));
    }

    return std::make_tuple(frame_ix, frame_weights, backward_sel, backward_weights);
}


#endif //MOVIE_UPSAMPLING_UPSAMPLE_MOVIE_H
