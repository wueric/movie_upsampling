#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define H_THREADS 32
#define W_THREADS 32


template<typename scalar_t>
__global__ void _kern_jitter_frames_forward(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> frames,
        const torch::PackedTensorAccessor<int64_t, 3, torch::RestrictPtrTraits, size_t> jitter_coords,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> jitter_dest
) {

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

    for (int64_t h = h_index; h < height; h += h_stride) {
        int64_t dest_h = jitter_h + h;
        for (int64_t w = w_index; w < width; w += w_stride) {
            int64_t dest_w = jitter_w + w;
            if (dest_h >= 0 && dest_h < height && dest_w >= 0 && dest_w < width) {
                jitter_dest[b][f][dest_h][dest_w] = frames[b][h][w];
            }
        }
    }
}


torch::Tensor _jitter_frames_forward(torch::Tensor frames,
                                    torch::Tensor jitter_coords) {
    /*
     *parallelize by height and width, since there will be many pixels, but probably not
     * very many frames
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
                frames.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                jitter_coords.packed_accessor<int64_t, 3, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}


template<typename scalar_t>
__global__ void _kern_jitter_frames_backward(
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_output_d_jittered_frames,
        const torch::PackedTensorAccessor<int64_t, 3, torch::RestrictPtrTraits, size_t> jitter_coords,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> d_output_d_static_frames) {

    int64_t batch = d_output_d_static_frames.size(0);

    int64_t n_frames = jitter_coords.size(1);

    int64_t b = blockIdx.x;

    int64_t height = d_output_d_static_frames.size(1);
    int64_t h_index = threadIdx.x;
    int64_t h_stride = blockDim.x;

    int64_t width = d_output_d_static_frames.size(2);
    int64_t w_index = threadIdx.y;
    int64_t w_stride = blockDim.y;

    for (int64_t f = 0; f < n_frames; ++f) {
        int64_t offset_h = jitter_coords[b][f][0];
        int64_t offset_w = jitter_coords[b][f][1];
        for (int64_t h = h_index; h < height; h += h_stride) {
            for (int64_t w = w_index; w < width; w += w_stride) {

                int64_t read_h = h + offset_h;
                int64_t read_w = w + offset_w;

                if (read_h >= 0 && read_h < height && read_w >= 0 && read_w < width) {
                    d_output_d_static_frames[b][h][w] += d_output_d_jittered_frames[b][f][read_h][read_w];
                }
            }

        }
    }
}


torch::Tensor _jitter_frames_backward(torch::Tensor d_output_d_jittered_frames,
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
    torch::Tensor dest = torch::zeros(std::vector<int64_t>({batch, height, width}), options);

    const dim3 threads(32, 32);
    const dim3 blocks(batch);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "_kern_jitter_frames_backward", [&] {
        _kern_jitter_frames_backward<scalar_t><<<blocks, threads>>>(
                d_output_d_jittered_frames.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                jitter_coords.packed_accessor<int64_t, 3, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}
