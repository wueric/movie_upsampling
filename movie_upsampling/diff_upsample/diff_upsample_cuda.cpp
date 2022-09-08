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
                                     torch::Tensor backward_weights){

    CHECK_INPUT(dloss_dflat_upsample);
    CHECK_INPUT(backward_selection);
    CHECK_INPUT(backward_weights);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(dloss_dflat_upsample));
    return _upsample_flat_forward(dloss_dflat_upsample, backward_selection, backward_weights);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("upsample_flat_forward", &upsample_flat_forward, "Upsample movie forward pass");
m.def("upsample_flat_backward", &upsample_flat_backward, "Upsample movie backward pass");
}
