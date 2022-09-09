//
// Created by Eric Wu on 6/13/22.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "upsample_movie.h"


PYBIND11_MODULE(upsampling_cpp_lib, m) {
m.doc() = "Fast temporal moving upsampling"; // optional module docstring

m.def("_compute_interval_overlaps",
    &_compute_interval_overlaps<float>,
    pybind11::return_value_policy::take_ownership,
    "Computes interval overlaps");

m.def("_batch_compute_interval_overlaps",
    &_batch_compute_interval_overlaps<float>,
    pybind11::return_value_policy::take_ownership,
    "Computes batched interval overlaps");

}

