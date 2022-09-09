//
// Created by Eric Wu on 9/1/22.
//

/*
 * We have two different functions here, both of which need to be differentiable
 * with respect to the stimulus image
 *
 *
 *
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor _jitter_frames_forward(torch::Tensor movie_frames,
                                     torch::Tensor jitter_coordinates);

torch::Tensor jitter_movie_forward(torch::Tensor movie_frames,
                                   torch::Tensor jitter_coordinates) {

    CHECK_INPUT(movie_frames);
    CHECK_INPUT(jitter_coordinates);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(movie_frames));
    return _jitter_frames_forward(movie_frames, jitter_coordinates);
}

torch::Tensor _jitter_frames_backward(torch::Tensor d_output_d_jittered_frames,
                                      torch::Tensor jitter_coords);


torch::Tensor jitter_movie_backward(torch::Tensor d_output_d_jittered_frames,
                                    torch::Tensor jitter_coordinates) {

    CHECK_INPUT(d_output_d_jittered_frames);
    CHECK_INPUT(jitter_coordinates);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(d_output_d_jittered_frames));
    return _jitter_frames_backward(d_output_d_jittered_frames, jitter_coordinates);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("jitter_movie_forward", &jitter_movie_forward, "Jitter movie forward pass");
m.def("jitter_movie_backward", &jitter_movie_backward, "Jitter movie backward pass");
}

