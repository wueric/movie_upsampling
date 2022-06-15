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

#include "NDArrayWrapper.h"

namespace py = pybind11;

template<class T>
using ContigNPArray = py::array_t<T, py::array::c_style | py::array::forcecast>;


void _raw_compute_interval_overlaps(CNDArrayWrapper::NDRawArrayWrapper<float, 1> movie_bin_cutoffs,
                                    CNDArrayWrapper::NDRawArrayWrapper<float, 1> spike_bin_cutoffs,
                                    CNDArrayWrapper::NDRawArrayWrapper<int64_t, 2> output_overlaps,
                                    CNDArrayWrapper::NDRawArrayWrapper<float, 2> frame_weights) {

    int64_t n_spike_bins = spike_bin_cutoffs.shape[0] - 1;
    int64_t n_frames = movie_bin_cutoffs.shape[0] - 1;

    int64_t frame_idx = 0;
    for (int64_t us_idx = 0; us_idx < n_spike_bins; ++us_idx) {
        float low = spike_bin_cutoffs.valueAt(us_idx);
        float high = spike_bin_cutoffs.valueAt(us_idx + 1);
        float bin_width = high - low;

        /* Determine which movie frames this interval overlaps with
         * Because this function is guaranteed to be upsampling movies
         * this interval is guaranteed to overlap with either one or two
         * movie frames
         */
        int64_t frame_low = frame_idx;
        while (frame_low < n_frames && movie_bin_cutoffs.valueAt(frame_low + 1) < low) ++frame_low;

        int64_t frame_high = frame_low;
        float curr_frame_start = movie_bin_cutoffs.valueAt(frame_high);
        float curr_frame_end = movie_bin_cutoffs.valueAt(frame_high + 1);

        if (curr_frame_start <= low && curr_frame_end >= high) {
            output_overlaps.storeTo(frame_high, us_idx, 0);
            frame_weights.storeTo(1.0, us_idx, 0);

            output_overlaps.storeTo(INVALID_FRAME, us_idx, 1);
            frame_weights.storeTo(0.0, us_idx, 1);

            frame_idx = frame_high;
        } else {

            float interval_overlap = (std::min(curr_frame_end, high) - std::max(curr_frame_start, low)) / bin_width;
            output_overlaps.storeTo(frame_high, us_idx, 0);
            frame_weights.storeTo(interval_overlap, us_idx, 0);

            ++frame_high;
            curr_frame_start = movie_bin_cutoffs.valueAt(frame_high);
            curr_frame_end = movie_bin_cutoffs.valueAt(frame_high + 1);
            interval_overlap = (std::min(curr_frame_end, high) - std::max(curr_frame_start, low)) / bin_width;
            output_overlaps.storeTo(frame_high, us_idx, 1);
            frame_weights.storeTo(interval_overlap, us_idx, 1);

            frame_idx = frame_high;
        }
    }
}


std::tuple <ContigNPArray<int64_t>, ContigNPArray<float>> _compute_interval_overlaps(
        ContigNPArray<float> movie_bin_cutoffs,
        ContigNPArray<float> spike_bin_cutoffs) {

    py::buffer_info movie_bin_info = movie_bin_cutoffs.request();
    auto *movie_bin_ptr = static_cast<float *>(movie_bin_info.ptr);
    const int64_t n_frame_cutoffs = movie_bin_info.shape[0];

    CNDArrayWrapper::NDRawArrayWrapper<float, 1> movie_bin_wrapper(
            movie_bin_ptr,
            {n_frame_cutoffs});

    py::buffer_info spike_bin_info = spike_bin_cutoffs.request();
    auto *spike_bin_ptr = static_cast<float *>(spike_bin_info.ptr);
    const int64_t n_bin_cutoffs = spike_bin_info.shape[0];
    const int64_t n_bins = n_bin_cutoffs - 1;

    CNDArrayWrapper::NDRawArrayWrapper<float, 1> spike_bin_wrapper(
            spike_bin_ptr,
            {n_bin_cutoffs});

    auto frame_weight_info = py::buffer_info(
            nullptr,
            sizeof(float),
            py::format_descriptor<float>::value,
            2, /* How many dimensions */
            {static_cast<py::ssize_t>(n_bins), static_cast<py::ssize_t>(2)}, /* shape */
            {static_cast<py::ssize_t>(2 * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))} /* stride */
    );
    ContigNPArray<float> frame_weights = ContigNPArray<float>(frame_weight_info);
    CNDArrayWrapper::NDRawArrayWrapper<float, 2> frame_weight_wrapper(static_cast<float *>(frame_weights.request().ptr),
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
    CNDArrayWrapper::NDRawArrayWrapper<int64_t, 2> frame_ix_wrapper(static_cast<int64_t *>(frame_ix.request().ptr),
                                                                    std::array<int64_t, 2>({n_bins, 2}));

    _raw_compute_interval_overlaps(movie_bin_wrapper,
                                   spike_bin_wrapper,
                                   frame_ix_wrapper,
                                   frame_weight_wrapper);

    return std::make_pair(frame_ix, frame_weights);

}

#endif //MOVIE_UPSAMPLING_UPSAMPLE_MOVIE_H
