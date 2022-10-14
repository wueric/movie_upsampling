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


torch::Tensor _beam_jitter_repeat_frames_forward(torch::Tensor movie_frames,
                                                 torch::Tensor beam_jitter_coords);


torch::Tensor beam_jitter_repeat_frames_forward(torch::Tensor movie_frames,
                                                torch::Tensor beam_jitter_coords) {
    /*
     * Not meant to be differentiable, so we don't care if inputs are non-contiguous
     *
     * @param frames: shape (beam, height, width)
     * @param beam_jitter_coords: shape (beam, beam_grid, n_jitter_frames, 2), int64_t
     */

    CHECK_CUDA(movie_frames);
    CHECK_CUDA(beam_jitter_coords);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(movie_frames));
    return _beam_jitter_repeat_frames_forward(movie_frames, beam_jitter_coords);
}


torch::Tensor _jitter_frames_only_forward(torch::Tensor repeated_frames,
                                          torch::Tensor jitter_coords);


torch::Tensor jitter_frames_only_forward(torch::Tensor repeated_frames,
                                         torch::Tensor jitter_coords) {

    CHECK_INPUT(repeated_frames);
    CHECK_INPUT(jitter_coords);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(repeated_frames));
    return _jitter_frames_only_forward(repeated_frames, jitter_coords);
}


torch::Tensor _jitter_frames_only_backward(torch::Tensor d_output_d_jittered_frames,
                                           torch::Tensor jitter_coords);


torch::Tensor jitter_frames_only_backward(torch::Tensor d_output_d_jittered_frames,
                                          torch::Tensor jitter_coords) {

    CHECK_INPUT(d_output_d_jittered_frames);
    CHECK_INPUT(jitter_coords);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(d_output_d_jittered_frames));
    return _jitter_frames_only_backward(d_output_d_jittered_frames, jitter_coords);
}


torch::Tensor _frame_repeat_forward(torch::Tensor frames,
                                    int64_t n_repeats);


torch::Tensor frame_repeat_forward(torch::Tensor frames,
                                   int64_t n_repeats) {

    CHECK_INPUT(frames);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(frames));
    return _frame_repeat_forward(frames, n_repeats);
}


torch::Tensor _frame_repeat_backward(torch::Tensor d_output_d_repeat_frames);


torch::Tensor frame_repeat_backward(torch::Tensor d_output_d_repeat_frames) {

    CHECK_INPUT(d_output_d_repeat_frames);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(d_output_d_repeat_frames));
    return _frame_repeat_backward(d_output_d_repeat_frames);
}


torch::Tensor _grid_jitter_single_frame_forward(
        torch::Tensor single_frame,
        torch::Tensor grid_jitter_coords);


torch::Tensor grid_jitter_single_frame_forward(
        torch::Tensor single_frame,
        torch::Tensor grid_jitter_coords) {

    CHECK_INPUT(single_frame);
    CHECK_INPUT(grid_jitter_coords);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(single_frame));

    return _grid_jitter_single_frame_forward(single_frame, grid_jitter_coords);
}


torch::Tensor _grid_jitter_single_frame_backward(
        torch::Tensor d_output_d_jittered_frames,
        torch::Tensor grid_jitter_coords);


torch::Tensor grid_jitter_single_frame_backward(
        torch::Tensor d_output_d_jittered_frames,
        torch::Tensor grid_jitter_coords) {

    CHECK_INPUT(d_output_d_jittered_frames);
    CHECK_INPUT(grid_jitter_coords);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(d_output_d_jittered_frames));

    return _grid_jitter_single_frame_backward(d_output_d_jittered_frames,
                                              grid_jitter_coords);
}


torch::Tensor _nobatch_grid_jitter_single_frame_forward(
        torch::Tensor single_frame,
        torch::Tensor grid_jitter_coords);


torch::Tensor nobatch_grid_jitter_single_frame_forward(
        torch::Tensor single_frame,
        torch::Tensor grid_jitter_coords) {

    CHECK_INPUT(single_frame);
    CHECK_INPUT(grid_jitter_coords);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(single_frame));
    return _nobatch_grid_jitter_single_frame_forward(single_frame, grid_jitter_coords);
}


torch::Tensor _nobatch_grid_jitter_single_frame_backward(
        torch::Tensor d_output_d_jittered_frames,
        torch::Tensor grid_jitter_coords);


torch::Tensor nobatch_grid_jitter_single_frame_backward(
        torch::Tensor d_output_d_jittered_frames,
        torch::Tensor grid_jitter_coords) {

    CHECK_INPUT(d_output_d_jittered_frames);
    CHECK_INPUT(grid_jitter_coords);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(d_output_d_jittered_frames));

    return _nobatch_grid_jitter_single_frame_backward(d_output_d_jittered_frames,
                                                      grid_jitter_coords);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("jitter_movie_forward", &jitter_movie_forward, "Jitter movie forward pass");
m.def("jitter_movie_backward", &jitter_movie_backward, "Jitter movie backward pass");
m.def("jitter_frames_only_forward", &jitter_frames_only_forward, "Jitter repeated frames forward pass");
m.def("jitter_frames_only_backward", &jitter_frames_only_backward, "Jitter repeated frames backward pass");
m.def("frame_repeat_forward", &frame_repeat_forward, "Repeat frames forward pass");
m.def("frame_repeat_backward", &frame_repeat_backward, "Repeat frames backward pass");
m.def("beam_jitter_repeat_frames_forward", &beam_jitter_repeat_frames_forward, "Beam search jitter repeat forward pass");
m.def("grid_jitter_single_frame_forward", &grid_jitter_single_frame_forward, "Forward pass for batched EM jitter");
m.def("grid_jitter_single_frame_backward", &grid_jitter_single_frame_backward, "Backward pass for batched EM jitter");
m.def("nobatch_grid_jitter_single_frame_forward", &nobatch_grid_jitter_single_frame_forward, "Forward pass for no-batch EM jitter");
m.def("nobatch_grid_jitter_single_frame_backward", &nobatch_grid_jitter_single_frame_backward, "Backward pass for no-batch EM jitter");
}

