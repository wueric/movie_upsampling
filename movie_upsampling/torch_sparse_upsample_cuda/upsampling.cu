#define INVALID_IDX -1

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

template<typename scalar_t>
__global__ void dumb_add_kernel(
        const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> a,
        const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> b,
        torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dest) {

    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const int64_t max_N = a.size(0);
    for (int64_t i = index; i < max_N; i += stride) {
        dest[i] = a[i] + b[i];
    }
}

torch::Tensor dumb_add_cuda(torch::Tensor a_tens,
                            torch::Tensor b_tens) {

    int64_t dim_a = a_tens.size(0);

    auto options = torch::TensorOptions()
            .dtype(a_tens.dtype())
            .layout(torch::kStrided)
            .device(a_tens.device());

    torch::Tensor dest = torch::empty(std::vector<int64_t>({dim_a}), options);

    const int threads = 1024;
    const dim3 blocks((dim_a + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(dest.scalar_type(), "dumb_add", [&] {
        dumb_add_kernel<scalar_t><<<blocks, threads>>>(
            a_tens.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            b_tens.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dest.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}

template<typename scalar_t>
__global__ void sparse_time_domain_movie_upsample_kernel(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> movie_frames,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> us_dest,
        const torch::PackedTensorAccessor<int64_t, 2, torch::RestrictPtrTraits, size_t> frame_selection,
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> frame_weights) {

    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    const int64_t max_N = us_dest.size(0);
    const int64_t height = us_dest.size(1);
    const int64_t width = us_dest.size(2);

    for (int64_t i = index; i < max_N; i += stride) {
        if (frame_selection[i][1] == INVALID_IDX) {
            int64_t only_frame_ix = frame_selection[i][0];
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    us_dest[i][h][w] = movie_frames[only_frame_ix][h][w];
                }
            }
        } else {

            int64_t first_frame_ix = frame_selection[i][0];
            scalar_t first_weight = frame_weights[i][0];

            int64_t second_frame_ix = frame_selection[i][1];;
            scalar_t second_weight = frame_weights[i][1];
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < height; ++w) {
                    scalar_t first_val = movie_frames[first_frame_ix][h][w];
                    scalar_t second_val = movie_frames[second_frame_ix][h][w];

                    scalar_t write_val = first_val * first_weight + second_val * second_weight;
                    us_dest[i][h][w] = write_val;
                }
            }
        }
    }
}

torch::Tensor upsample_sparse_movie_cuda(torch::Tensor movie_frames,
                                         torch::Tensor frame_selection,
                                         torch::Tensor frame_weights) {
    int64_t n_bins = frame_selection.size(0);
    int64_t height = movie_frames.size(1);
    int64_t width = movie_frames.size(2);

    auto options = torch::TensorOptions()
            .dtype(frame_weights.dtype())
            .layout(torch::kStrided)
            .device(movie_frames.device());

    torch::Tensor dest = torch::empty(std::vector<int64_t>({n_bins, height, width}), options);

    const int threads = 1024;
    const dim3 blocks((n_bins + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(dest.scalar_type(), "sparse_upsample_movie", [&] {
        sparse_time_domain_movie_upsample_kernel<scalar_t><<<blocks, threads>>>(
                movie_frames.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                frame_selection.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
                frame_weights.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}
