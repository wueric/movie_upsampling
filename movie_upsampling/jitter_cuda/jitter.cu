#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>


template<typename scalar_t>
__global__ void _kern_frame_repeat_forward(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> frames,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> repeat_dest) {

    int64_t b = threadIdx.x + blockDim.x * blockIdx.x;

    int64_t h_index = threadIdx.y + blockDim.y * blockIdx.y;
    int64_t h_stride = blockDim.y * gridDim.y;

    int64_t w_index = threadIdx.z + blockDim.z * blockIdx.z;
    int64_t w_stride = blockDim.z * gridDim.z;

    const int64_t n_frames = repeat_dest.size(1);
    const int64_t height = frames.size(1);
    const int64_t width = frames.size(2);

    for (int64_t h = h_index; h < height; h += h_stride) {
        for (int64_t w = w_index; w < width; w += w_stride) {

            scalar_t to_repeat = frames[b][h][w];
            for (int64_t f = 0; f < n_frames; ++f) {
                repeat_dest[b][f][h][w] = to_repeat;
            }
        }
    }
}


torch::Tensor _frame_repeat_forward(torch::Tensor frames,
                                    int64_t n_repeats) {

    const int64_t batch = frames.size(0);
    const int64_t height = frames.size(1);
    const int64_t width = frames.size(2);

    auto options = torch::TensorOptions()
            .dtype(frames.dtype())
            .layout(torch::kStrided)
            .device(frames.device());
    torch::Tensor dest = torch::zeros(std::vector<int64_t>({batch, n_repeats, height, width}), options);

    const dim3 threads(1, 32, 32);
    const dim3 blocks(batch, 4, 4);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "_kern_frame_repeat_forward", [&] {
        _kern_frame_repeat_forward<scalar_t><<<blocks, threads>>>(
                frames.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}


template<typename scalar_t>
__global__ void _kern_frame_repeat_backward(
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_output_d_repeat_frames,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> d_output_d_static_frames) {

    int64_t batch = d_output_d_static_frames.size(0);
    int64_t n_frames = d_output_d_repeat_frames.size(1);

    int64_t b = threadIdx.x + blockIdx.x * blockDim.x;

    int64_t height = d_output_d_repeat_frames.size(2);
    int64_t h_index = threadIdx.y + blockDim.y * blockIdx.y;
    int64_t h_stride = blockDim.y * gridDim.y;

    int64_t width = d_output_d_repeat_frames.size(3);
    int64_t w_index = threadIdx.z + blockDim.z * blockIdx.z;
    int64_t w_stride = blockDim.z * gridDim.z;

    for (int64_t h = h_index; h < height; h += h_stride) {
        for (int64_t w = w_index; w < width; w += w_stride) {

            scalar_t acc = 0.0;
            for (int64_t f = 0; f < n_frames; ++f) {
                acc += d_output_d_repeat_frames[b][f][h][w];
            }

            d_output_d_static_frames[b][h][w] = acc;
        }
    }
}


torch::Tensor _frame_repeat_backward(torch::Tensor d_output_d_repeat_frames) {
    /*
     * Computes the gradient of the loss with respect to the input
     */

    int64_t batch = d_output_d_repeat_frames.size(0);
    int64_t n_jittered_frames = d_output_d_repeat_frames.size(1);
    int64_t height = d_output_d_repeat_frames.size(2);
    int64_t width = d_output_d_repeat_frames.size(3);

    auto options = torch::TensorOptions()
            .dtype(d_output_d_repeat_frames.dtype())
            .layout(torch::kStrided)
            .device(d_output_d_repeat_frames.device());
    torch::Tensor dest = torch::zeros(std::vector<int64_t>({batch, height, width}), options);

    const dim3 threads(1, 32, 32);
    const dim3 blocks(batch, 4, 4);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "_kern_frame_repeat_backward", [&] {
        _kern_frame_repeat_backward<scalar_t><<<blocks, threads>>>(
                d_output_d_repeat_frames.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}

template<typename scalar_t>
__global__ void _kern_jitter_only_forward(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> repeated_frames,
        const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> jitter_coords,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> jitter_dest) {

    const int64_t height = repeated_frames.size(2);
    int64_t h_index = threadIdx.x;
    int64_t h_stride = blockDim.x;

    const int64_t width = repeated_frames.size(3);
    int64_t w_index = threadIdx.y;
    int64_t w_stride = blockDim.y;

    int64_t b = blockIdx.x;
    int64_t f = blockIdx.y;

    int64_t jitter_h = jitter_coords[b][f][0];
    int64_t jitter_w = jitter_coords[b][f][1];

    scalar_t ZERO = 0.0;

    for (int64_t h = h_index; h < height; h += h_stride) {
        int64_t source_h = h - jitter_h;
        bool valid_source_h = (source_h >= 0) && (source_h < height);

        for (int64_t w = w_index; w < width; w += w_stride) {
            int64_t source_w = w - jitter_w;
            bool valid_source_w = (source_w >= 0) && (source_w < width);
            jitter_dest[b][f][h][w] = (valid_source_h && valid_source_w) ? repeated_frames[b][f][source_h][source_w]
                                                                         : ZERO;
        }
    }
}


torch::Tensor _jitter_frames_only_forward(torch::Tensor repeated_frames,
                                          torch::Tensor jitter_coords) {
    /*
     *parallelize by height and width, since there will be many pixels, but probably not
     * very many frames
     */

    const int64_t batch = repeated_frames.size(0);
    const int64_t height = repeated_frames.size(2);
    const int64_t width = repeated_frames.size(3);
    const int64_t n_jitter_frames = jitter_coords.size(1);

    auto options = torch::TensorOptions()
            .dtype(repeated_frames.dtype())
            .layout(torch::kStrided)
            .device(repeated_frames.device());
    torch::Tensor dest = torch::zeros(std::vector<int64_t>({batch, n_jitter_frames, height, width}), options);

    const dim3 threads(32, 32);
    const dim3 blocks(batch, n_jitter_frames);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "_kern_jitter_only_forward", [&] {
        _kern_jitter_only_forward<scalar_t><<<blocks, threads>>>(
                repeated_frames.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                jitter_coords.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                dest.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
    });

    return dest;
}


template<typename scalar_t>
__global__ void _kern_jitter_frames_only_backward(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_output_d_jittered_frames,
        const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> jitter_coords,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_output_d_repeat_frames) {
    /*
     * @param d_output_d_jittered_frames: shape (batch, n_jittered_frames, height, width)
     * @param jitter_coords: shape (batch, n_jittered_frames, 2)
     *
     * returns: shape (batch, n_jittered_frames, height, width)
     */

    const int64_t height = d_output_d_jittered_frames.size(2);
    int64_t h_index = threadIdx.x;
    int64_t h_stride = blockDim.x;

    const int64_t width = d_output_d_jittered_frames.size(3);
    int64_t w_index = threadIdx.y;
    int64_t w_stride = blockDim.y;

    int64_t b = blockIdx.x;
    int64_t f = blockIdx.y;

    int64_t jitter_h = jitter_coords[b][f][0];
    int64_t jitter_w = jitter_coords[b][f][1];

    scalar_t ZERO = 0.0;

    for (int64_t h = h_index; h < height; h += h_stride) {
        int64_t read_h = h + jitter_h;
        bool sat_h = (read_h >= 0) && (read_h < height);

        for (int64_t w = w_index; w < width; w += w_stride) {

            int64_t read_w = w + jitter_w;
            bool sat_w = (read_w >= 0) && (read_w < width);

            scalar_t add_to = (sat_h && sat_w) ? d_output_d_jittered_frames[b][f][read_h][read_w] : ZERO;
            d_output_d_repeat_frames[b][f][h][w] = add_to;
        }
    }
}


torch::Tensor _jitter_frames_only_backward(torch::Tensor d_output_d_jittered_frames,
                                           torch::Tensor jitter_coords) {
    /*
     * Computes the gradient of the loss with respect to the input
     */

    int64_t batch = d_output_d_jittered_frames.size(0);
    int64_t n_jittered_frames = d_output_d_jittered_frames.size(1);
    int64_t height = d_output_d_jittered_frames.size(2);
    int64_t width = d_output_d_jittered_frames.size(3);

    auto options = torch::TensorOptions()
            .dtype(d_output_d_jittered_frames.dtype())
            .layout(torch::kStrided)
            .device(d_output_d_jittered_frames.device());
    torch::Tensor dest = torch::zeros(std::vector<int64_t>({batch, n_jittered_frames, height, width}), options);

    const dim3 threads(32, 32);
    const dim3 blocks(batch, n_jittered_frames);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "_kern_jitter_frames_only_backward", [&] {
        _kern_jitter_frames_only_backward<scalar_t><<<blocks, threads>>>(
                d_output_d_jittered_frames.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                jitter_coords.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                dest.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
    });

    return dest;
}


template<typename scalar_t>
__global__ void _kern_jitter_frames_forward(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> frames,
        const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> jitter_coords,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> jitter_dest) {

    const int64_t height = frames.size(1);
    int64_t h_index = threadIdx.x;
    int64_t h_stride = blockDim.x;

    const int64_t width = frames.size(2);
    int64_t w_index = threadIdx.y;
    int64_t w_stride = blockDim.y;

    int64_t b = blockIdx.x;
    int64_t f = blockIdx.y;

    int64_t jitter_h = jitter_coords[b][f][0];
    int64_t jitter_w = jitter_coords[b][f][1];

    scalar_t ZERO = 0.0;
    for (int64_t h = h_index; h < height; h += h_stride) {
        int64_t source_h = h - jitter_h;
        bool valid_source_h = (source_h >= 0) && (source_h < height);

        for (int64_t w = w_index; w < width; w += w_stride) {
            int64_t source_w = w - jitter_w;
            bool valid_source_w = (source_w >= 0) && (source_w < width);

            jitter_dest[b][f][h][w] = (valid_source_h && valid_source_w) ?
                                      frames[b][source_h][source_w] : ZERO;
        }
    }
}


torch::Tensor _jitter_frames_forward(
        torch::Tensor frames,
        torch::Tensor jitter_coords) {
    /*
     *parallelize by height and width, since there will be many pixels, but probably not
     * very many frames
     *
     * @param frames: shape (batch, height, width)
     * @param jitter_coords: (batch, n_jittered_frames, 2)
     */

    const int64_t batch = frames.size(0);
    const int64_t height = frames.size(1);
    const int64_t width = frames.size(2);

    const int64_t n_jitter_frames = jitter_coords.size(1);

    auto options = torch::TensorOptions()
            .dtype(frames.dtype())
            .layout(torch::kStrided)
            .device(frames.device());
    torch::Tensor dest = torch::zeros(std::vector<int64_t>({batch, n_jitter_frames, height, width}), options);

    const dim3 threads(32, 32);
    const dim3 blocks(batch, n_jitter_frames);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "_kern_jitter_frames_forward", [&] {
        _kern_jitter_frames_forward<scalar_t><<<blocks, threads>>>(
                frames.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                jitter_coords.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                dest.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
    });

    return dest;
}


torch::Tensor _jitter_frames_backward(torch::Tensor d_output_d_jittered_frames,
                                      torch::Tensor jitter_coords) {
    /*
     * Computes the gradient of the loss with respect to the input
     *
     * This function will only be used for reconstruction, where the total number
     * of images being reconstructed in a single batch is small (~8 at most)
     *
     * This means that we can get away with 32-bit indexing everywhere, since none
     * of the tensors will be particularly large
     *
     * @param d_output_d_jittered_frames: shape (batch, n_jittered_frames, height, width)
     * @param jitter_coords: shape (batch, n_jittered_frames, 2)
     */

    int64_t batch = d_output_d_jittered_frames.size(0);
    int64_t n_jittered_frames = d_output_d_jittered_frames.size(1);
    int64_t height = d_output_d_jittered_frames.size(2);
    int64_t width = d_output_d_jittered_frames.size(3);

    auto options = torch::TensorOptions()
            .dtype(d_output_d_jittered_frames.dtype())
            .layout(torch::kStrided)
            .device(d_output_d_jittered_frames.device());
    torch::Tensor d_unjittered = torch::zeros(std::vector<int64_t>({batch, n_jittered_frames, height, width}),
                                              options);

    const dim3 threads(32, 32);
    const dim3 blocks(batch, n_jittered_frames);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(d_unjittered.scalar_type(), "_kern_jitter_frames_only_backward", [&] {
        _kern_jitter_frames_only_backward<scalar_t><<<blocks, threads>>>(
                d_output_d_jittered_frames.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                jitter_coords.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                d_unjittered.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
    });

    return d_unjittered.sum(1);
}


template<typename scalar_t>
__global__ void _kern_beam_jitter_repeat_frames_forward(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> frames,
        const torch::PackedTensorAccessor<int64_t, 4, torch::RestrictPtrTraits, size_t> jitter_coords,
        torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> jitter_dest) {

    const int64_t height = frames.size(1);
    int64_t h_index = threadIdx.x;
    int64_t h_stride = blockDim.x;

    const int64_t width = frames.size(2);
    int64_t w_index = threadIdx.y;
    int64_t w_stride = blockDim.y;

    int64_t beam_idx = blockIdx.x;
    int64_t beam_grid_idx = blockIdx.y;
    int64_t frame_idx = blockIdx.z;

    int64_t jitter_h = jitter_coords[beam_idx][beam_grid_idx][frame_idx][0];
    int64_t jitter_w = jitter_coords[beam_idx][beam_grid_idx][frame_idx][1];

    scalar_t ZERO = 0.0;
    for (int64_t h = h_index; h < height; h += h_stride) {
        int64_t source_h = h - jitter_h;
        bool valid_source_h = (source_h >= 0) && (source_h < height);

        for (int64_t w = w_index; w < width; w += w_stride) {
            int64_t source_w = w - jitter_w;
            bool valid_source_w = (source_w >= 0) && (source_w < width);

            jitter_dest[beam_idx][beam_grid_idx][frame_idx][h][w] = (valid_source_h && valid_source_w) ?
                                                                    frames[beam_idx][source_h][source_w] : ZERO;

        }
    }
}


torch::Tensor _beam_jitter_repeat_frames_forward(
        torch::Tensor frames,
        torch::Tensor beam_jitter_coords) {
    /*
     * This operation is not meant to be differentiable, so it is not paired
     * with a backwards method
     *
     * @param frames: shape (beam, height, width)
     * @param beam_jitter_coords: shape (beam, beam_grid, n_jitter_frames, 2), int64_t
     */

    const int64_t beam = frames.size(0);
    const int64_t height = frames.size(1);
    const int64_t width = frames.size(2);

    const int64_t beam_grid = beam_jitter_coords.size(1);
    const int64_t n_jitter_frames = beam_jitter_coords.size(2);

    auto options = torch::TensorOptions()
            .dtype(frames.dtype())
            .layout(torch::kStrided)
            .device(frames.device());
    torch::Tensor dest = torch::zeros(std::vector<int64_t>({beam, beam_grid, n_jitter_frames, height, width}),
                                      options);

    const dim3 threads(16, 32);
    const dim3 blocks(beam, beam_grid, n_jitter_frames);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "_kern_beam_jitter_repeat_frames_forward", [&] {
        _kern_beam_jitter_repeat_frames_forward<scalar_t><<<blocks, threads>>>(
                frames.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                beam_jitter_coords.packed_accessor<int64_t, 4, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}


template<typename scalar_t>
__global__ void _kern_grid_jitter_single_frame_forward(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> frame,
        const torch::PackedTensorAccessor32<int64_t, 4, torch::RestrictPtrTraits> jitter_coord,
        torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> jitter_dest) {
    /*
     * @param frame: shape (batch, height, width)
     * @param jitter_coord: shape (batch, n_grid, n_jittered_frames, 2)
     * @param jitter_dest: shape (batch, n_grid, n_jitter_frames, height, width)
     *                      OUTPUT
     */

    const int64_t height = frame.size(1);
    int64_t h_index = threadIdx.x;
    int64_t h_stride = blockDim.x;

    const int64_t width = frame.size(2);
    int64_t w_index = threadIdx.y;
    int64_t w_stride = blockDim.y;

    int64_t b = blockIdx.x;
    int64_t g = blockIdx.y;
    int64_t f = blockIdx.z;

    int64_t jitter_h = jitter_coord[b][g][f][0];
    int64_t jitter_w = jitter_coord[b][g][f][1];

    scalar_t ZERO = 0.0;
    for (int64_t h = h_index; h < height; h += h_stride) {
        int64_t source_h = h - jitter_h;
        bool valid_source_h = (source_h >= 0) && (source_h < height);

        for (int64_t w = w_index; w < width; w += w_stride) {
            int64_t source_w = w - jitter_w;
            bool valid_source_w = (source_w >= 0) && (source_w < width);

            jitter_dest[b][g][f][h][w] = (valid_source_h && valid_source_w) ?
                                         frame[b][source_h][source_w] : ZERO;
        }
    }
}


torch::Tensor _grid_jitter_single_frame_forward(
        torch::Tensor single_frame,
        torch::Tensor grid_jitter_coords) {
    /*
     * @param single_frame: shape (batch, height, width)
     * @param grid_jitter_coords: shape (batch, n_grid, n_jittered_frames, 2)
     *
     * returns: shape (batch, n_grid, n_jittered_frames, height, width)
     */

    const int64_t batch = single_frame.size(0);
    const int64_t height = single_frame.size(1);
    const int64_t width = single_frame.size(2);

    const int64_t n_grid = grid_jitter_coords.size(1);
    const int64_t n_jitter_frames = grid_jitter_coords.size(2);;

    auto options = torch::TensorOptions()
            .dtype(single_frame.dtype())
            .layout(torch::kStrided)
            .device(single_frame.device());
    torch::Tensor dest = torch::zeros(std::vector<int64_t>({batch, n_grid, n_jitter_frames, height, width}),
                                      options);

    const dim3 threads(32, 32);
    const dim3 blocks(batch, n_grid, n_jitter_frames);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "_kern_grid_jitter_single_frame_forward", [&] {
        _kern_grid_jitter_single_frame_forward<scalar_t><<<blocks, threads>>>(
                single_frame.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                grid_jitter_coords.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>(),
                dest.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>());
    });

    return dest;
}


template<typename scalar_t>
__global__ void _kern_grid_jitter_single_frame_backward_noreduce_frames(
        const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> d_output_d_jittered_frames,
        const torch::PackedTensorAccessor32<int64_t, 4, torch::RestrictPtrTraits> jitter_coords,
        torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> d_output_d_static_noreduce_frames) {

    /*
     * @param d_output_d_jittered_frames: shape (batch, n_grid, n_jittered_frames, height, width)
     * @param jitter_coords: shape (batch, n_grid, n_jittered_frames, 2)
     *
     * output: shape (batch, n_grid, n_jittered_frames, height, width)
     */


    const int64_t batch = d_output_d_jittered_frames.size(0);
    const int64_t n_grid = d_output_d_jittered_frames.size(1);
    const int64_t n_jittered_frames = d_output_d_jittered_frames.size(2);

    const int64_t height = d_output_d_jittered_frames.size(3);
    int64_t h_index = threadIdx.x;
    int64_t h_stride = blockDim.x;

    const int64_t width = d_output_d_jittered_frames.size(4);
    int64_t w_index = threadIdx.y;
    int64_t w_stride = blockDim.y;

    int64_t f = blockIdx.x;
    int64_t g = blockIdx.y;
    int64_t b = blockIdx.z;

    int64_t jitter_h = jitter_coords[b][g][f][0];
    int64_t jitter_w = jitter_coords[b][g][f][1];

    scalar_t ZERO = 0.0;

    for (int64_t h = h_index; h < height; h += h_stride) {
        int64_t read_h = h + jitter_h;
        bool sat_h = (read_h >= 0) && (read_h < height);

        for (int64_t w = w_index; w < width; w += w_stride) {

            int64_t read_w = w + jitter_w;
            bool sat_w = (read_w >= 0) && (read_w < width);

            scalar_t d_unjittered = (sat_h && sat_w) ? d_output_d_jittered_frames[b][g][f][read_h][read_w] : ZERO;
            d_output_d_static_noreduce_frames[b][g][f][h][w] = d_unjittered;
        }
    }
}


torch::Tensor _grid_jitter_single_frame_backward(
        torch::Tensor d_output_d_jittered_frames,
        torch::Tensor grid_jitter_coords) {
    /*
     * @param d_output_d_jittered_frames: shape (batch, n_grid, n_jittered_frames, height, width)
     * @param grid_jitter_coords: shape (batch, n_grid, n_jittered_frames, 2)
     *
     * returns: shape (batch, height, width)
     */

    const int64_t batch = d_output_d_jittered_frames.size(0);
    const int64_t n_grid = d_output_d_jittered_frames.size(1);
    const int64_t n_jittered_frames = d_output_d_jittered_frames.size(2);
    const int64_t height = d_output_d_jittered_frames.size(3);
    const int64_t width = d_output_d_jittered_frames.size(4);

    auto options = torch::TensorOptions()
            .dtype(d_output_d_jittered_frames.dtype())
            .layout(torch::kStrided)
            .device(d_output_d_jittered_frames.device());

    torch::Tensor d_unjittered = torch::zeros(std::vector<int64_t>({batch, n_grid, n_jittered_frames, height, width}),
                                              options);

    const dim3 threads(32, 32);
    const dim3 blocks(n_jittered_frames, n_grid, batch);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(d_unjittered.scalar_type(),
                                        "_kern_grid_jitter_single_frame_backward_noreduce_frames", [&] {
                _kern_grid_jitter_single_frame_backward_noreduce_frames<scalar_t><<<blocks, threads>>>(
                        d_output_d_jittered_frames.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                        grid_jitter_coords.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>(),
                        d_unjittered.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>());
            });

    // now we have to do a reduction over dest to sum over the grid dimension
    // and the frame dimension
    return d_unjittered.sum(2).sum(1);
}