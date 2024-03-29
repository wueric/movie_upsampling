//
// Created by Eric Wu on 9/8/22.
//

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor _upsample_flat_forward(torch::Tensor flat_noupsample,
                                     torch::Tensor flat_selection,
                                     torch::Tensor flat_weights);


torch::Tensor upsample_flat_forward(torch::Tensor flat_noupsample,
                                    torch::Tensor flat_selection,
                                    torch::Tensor flat_weights) {
    CHECK_INPUT(flat_noupsample);
    CHECK_INPUT(flat_selection);
    CHECK_INPUT(flat_weights);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(flat_noupsample));
    return _upsample_flat_forward(flat_noupsample, flat_selection, flat_weights);
}


torch::Tensor _upsample_flat_backward(torch::Tensor dloss_dflat_upsample,
                                      torch::Tensor backward_selection,
                                      torch::Tensor backward_weights);


torch::Tensor upsample_flat_backward(torch::Tensor dloss_dflat_upsample,
                                     torch::Tensor backward_selection,
                                     torch::Tensor backward_weights) {

    CHECK_INPUT(dloss_dflat_upsample);
    CHECK_INPUT(backward_selection);
    CHECK_INPUT(backward_weights);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(dloss_dflat_upsample));
    return _upsample_flat_backward(dloss_dflat_upsample, backward_selection, backward_weights);
}


torch::Tensor _upsample_transpose_flat_forward(torch::Tensor flat_noupsample,
                                               torch::Tensor flat_selection,
                                               torch::Tensor flat_weights);


torch::Tensor upsample_transpose_flat_forward(torch::Tensor flat_noupsample,
                                              torch::Tensor flat_selection,
                                              torch::Tensor flat_weights) {
    CHECK_INPUT(flat_noupsample);
    CHECK_CUDA(flat_selection);
    CHECK_CUDA(flat_weights);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(flat_noupsample));
    return _upsample_transpose_flat_forward(flat_noupsample, flat_selection, flat_weights);
}


torch::Tensor _upsample_transpose_flat_backward(torch::Tensor dloss_dflat_upsample,
                                                torch::Tensor backward_selection,
                                                torch::Tensor backward_weights);


torch::Tensor upsample_transpose_flat_backward(torch::Tensor dloss_dflat_upsample,
                                               torch::Tensor backward_selection,
                                               torch::Tensor backward_weights) {
    CHECK_INPUT(dloss_dflat_upsample);
    CHECK_CUDA(backward_selection);
    CHECK_CUDA(backward_weights);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(dloss_dflat_upsample));
    return _upsample_transpose_flat_backward(dloss_dflat_upsample, backward_selection, backward_weights);
}


torch::Tensor _shared_clock_time_upsample_transpose_forward(torch::Tensor flat_noupsample,
                                                            torch::Tensor flat_selection,
                                                            torch::Tensor flat_weights);


torch::Tensor shared_clock_time_upsample_transpose_forward(torch::Tensor flat_noupsample,
                                                           torch::Tensor flat_selection,
                                                           torch::Tensor flat_weights) {
    CHECK_INPUT(flat_noupsample);
    CHECK_INPUT(flat_selection);
    CHECK_INPUT(flat_weights);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(flat_noupsample));
    return _shared_clock_time_upsample_transpose_forward(flat_noupsample, flat_selection, flat_weights);
}

torch::Tensor _shared_clock_upsample_transpose_flat_backward(torch::Tensor dloss_dflat_upsample,
                                                             torch::Tensor backward_selection,
                                                             torch::Tensor backward_weights);


torch::Tensor shared_clock_upsample_transpose_flat_backward(torch::Tensor dloss_dflat_upsample,
                                                            torch::Tensor backward_selection,
                                                            torch::Tensor backward_weights) {
    CHECK_INPUT(dloss_dflat_upsample);
    CHECK_INPUT(backward_selection);
    CHECK_INPUT(backward_weights);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(dloss_dflat_upsample));
    return _shared_clock_upsample_transpose_flat_backward(dloss_dflat_upsample, backward_selection, backward_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("upsample_flat_forward", &upsample_flat_forward, "Upsample movie forward pass");
m.def("upsample_flat_backward", &upsample_flat_backward, "Upsample movie backward pass");
m.def("upsample_transpose_flat_forward", &upsample_transpose_flat_forward, "Upsample movie forward pass");
m.def("upsample_transpose_flat_backward", &upsample_transpose_flat_backward, "Upsample movie backward pass");
m.def("shared_clock_time_upsample_transpose_forward", &shared_clock_time_upsample_transpose_forward,
"Upsample movie forward pass with single shared clock");
m.def("shared_clock_upsample_transpose_flat_backward", &shared_clock_upsample_transpose_flat_backward,
      "Upsample movie backward pass with single shared clock");
}
