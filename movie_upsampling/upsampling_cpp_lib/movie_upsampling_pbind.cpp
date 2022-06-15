//
// Created by Eric Wu on 6/13/22.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "upsample_movie.h"


PYBIND11_MODULE(upsampling_cpp_lib, m) {
m.doc() = "Fast temporal moving upsampling"; // optional module docstring

m.def("compute_interval_overlaps",
&compute_interval_overlaps,
pybind11::return_value_policy::take_ownership,
"Computes interval overlaps");

}

