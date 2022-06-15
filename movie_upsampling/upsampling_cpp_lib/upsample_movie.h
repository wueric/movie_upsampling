//
// Created by Eric Wu on 6/13/22.
//

#ifndef MOVIE_UPSAMPLING_UPSAMPLE_MOVIE_H
#define MOVIE_UPSAMPLING_UPSAMPLE_MOVIE_H

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstdint>
#include <queue>
#include <memory>

#include "NDArrayWrapper.h"

namespace py = pybind11;

template<class T>
using ContigNPArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template<typename T, typename U>
void multiply_accumulate_2D_buffer(CNDArrayWrapper::StaticNDArrayWrapper<T, 2> &buffer,
                                   CNDArrayWrapper::StaticNDArrayWrapper<U, 2> &read_from,
                                   T mul_by) {

    int64_t n_rows = buffer.shape[0];
    int64_t n_cols = buffer.shape[1];

    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            T write_val = mul_by * (static_cast<T> (read_from.template valueAt(i, j)));
            buffer.storeTo(write_val, i, j);
        }
    }
}

template<typename T, typename U>
void copy_2D_buffer(CNDArrayWrapper::StaticNDArrayWrapper<T, 2> &buffer,
                    CNDArrayWrapper::StaticNDArrayWrapper<U, 2> &copy_from) {

    int64_t n_rows = buffer.shape[0];
    int64_t n_cols = buffer.shape[1];

    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            T write_val = static_cast<T>(copy_from.template valueAt(i, j));
            buffer.storeTo(write_val, i, j);
        }
    }
}

template<typename T>
void zero_2D_buffer(CNDArrayWrapper::StaticNDArrayWrapper<T, 2> &buffer) {
    int64_t n_rows = buffer.shape[0];
    int64_t n_cols = buffer.shape[1];

    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            buffer.storeTo(0, i, j);
        }
    }
}

ContigNPArray<float> temporal_upsample_transpose_movie(
        ContigNPArray<uint8_t> movie_frames,
        ContigNPArray<float> movie_bin_cutoffs,
        ContigNPArray<float> spike_bin_cutoffs) {

    py::buffer_info movie_frame_info = movie_frames.request();
    auto *movie_frame_ptr = static_cast<uint8_t *>(movie_frame_info.ptr);
    const int64_t n_frames = movie_frame_info.shape[0];
    const int64_t height = movie_frame_info.shape[1];
    const int64_t width = movie_frame_info.shape[2];

    CNDArrayWrapper::StaticNDArrayWrapper<uint8_t, 3> movie_frame_wrapper(
            movie_frame_ptr,
            {n_frames, height, width}
    );

    py::buffer_info spike_bin_info = spike_bin_cutoffs.request();
    auto *spike_bin_ptr = static_cast<float *>(spike_bin_info.ptr);
    const int64_t n_bin_cutoffs = spike_bin_info.shape[0];
    const int64_t n_bins = n_bin_cutoffs - 1;

    CNDArrayWrapper::StaticNDArrayWrapper<float, 1> spike_bin_wrapper(
            spike_bin_ptr,
            {n_bin_cutoffs}
    );

    py::buffer_info movie_bin_info = movie_bin_cutoffs.request();
    auto *movie_bin_ptr = static_cast<float *>(movie_bin_info.ptr);
    const int64_t n_frame_cutoffs = movie_bin_info.shape[0];

    CNDArrayWrapper::StaticNDArrayWrapper<float, 1> movie_bin_wrapper(
            movie_bin_ptr,
            {n_frame_cutoffs}
    );

    // allocate memory for the upsampled movie
    auto upsampled_movie_info = py::buffer_info(
            nullptr,
            sizeof(float),
            py::format_descriptor<float>::value,
            3, /* How many dimensions */
            {n_bins, height, width}, /* shape */
            {sizeof(float) * height * width, sizeof(float) * width, sizeof(float)} /* stride */
    );

    ContigNPArray<float> upsampled_movie = ContigNPArray<float>(upsampled_movie_info);
    //memset(upsampled_movie.request().ptr, 0, height * width * n_bins * sizeof(float));

    // and also make a wrapper for the upsampled movie
    CNDArrayWrapper::StaticNDArrayWrapper<float, 3> upsampled_wrapper(
            static_cast<float *>(upsampled_movie.request().ptr),
            {n_bins, height, width}
    );

    int64_t frame_idx = 0;
    for (int64_t us_idx = 0; us_idx < n_bins; ++us_idx) {

        CNDArrayWrapper::StaticNDArrayWrapper<float, 2> upsampled_slice_wrapper = upsampled_wrapper.slice<2>(
                CNDArrayWrapper::makeIdxSlice(us_idx),
                CNDArrayWrapper::makeAllSlice(),
                CNDArrayWrapper::makeAllSlice()
        );

        float low = spike_bin_wrapper.valueAt(us_idx);
        float high = spike_bin_wrapper.valueAt(us_idx + 1);
        float bin_width = high - low;

        /* Determine which movie frames this interval overlaps with
         * Because this function is guaranteed to be upsampling movies
         * this interval is guaranteed to overlap with either one or two
         * movie frames
         */
        int64_t frame_low = frame_idx;
        while (frame_low < n_frames && movie_bin_wrapper.valueAt(frame_low + 1) < low) ++frame_low;

        int64_t frame_high = frame_low - 1;
        while (frame_high < (n_frames - 1) && movie_bin_wrapper.valueAt(frame_high + 1) < high) {
            ++frame_high;

            float curr_frame_start = movie_bin_wrapper.valueAt(frame_high);
            float curr_frame_end = movie_bin_wrapper.valueAt(frame_high + 1);

            float interval_overlap = std::min(curr_frame_end, high) - std::max(curr_frame_start, low);

            CNDArrayWrapper::StaticNDArrayWrapper<uint8_t, 2> frame_slice_wrapper = movie_frame_wrapper.slice<2>(
                    CNDArrayWrapper::makeIdxSlice(frame_high),
                    CNDArrayWrapper::makeAllSlice(),
                    CNDArrayWrapper::makeAllSlice()
            );

            /*
             * For fast performance, we should avoid doing the multiplication where possible and just copy memory
             * in the case that the current spike time bin is covered by only 1 frame.
             * There are two cases, either (a) interval_overlap < bin_width, OR (b) interval_overlap = bin_width
             *
             * Implementation below is very sketchy but probably correct; see if we can find a better
             * way to do the copy outside of the loop
             */


            if (interval_overlap < bin_width) {
                float overlap_fraction = interval_overlap / bin_width;
                multiply_accumulate_2D_buffer<float, uint8_t>(upsampled_slice_wrapper,
                                                              frame_slice_wrapper,
                                                              overlap_fraction);

            } else {
                copy_2D_buffer<float, uint8_t>(upsampled_slice_wrapper,
                                               frame_slice_wrapper);
                break;
            }
        }

        frame_idx = frame_high;
    }

    return upsampled_movie;
}


#endif //MOVIE_UPSAMPLING_UPSAMPLE_MOVIE_H
