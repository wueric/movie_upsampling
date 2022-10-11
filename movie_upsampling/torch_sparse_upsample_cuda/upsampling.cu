#define INVALID_IDX -1

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template<typename scalar_t>
__global__ void dumb_add_kernel(
        const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> a,
        const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> b,
        torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dest) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

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

    torch::Tensor dest = torch::ones(std::vector<int64_t>({dim_a}), options);

    const int threads = 1024;
    const dim3 blocks((dim_a + threads - 1) / threads) ;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "dumb_add", ([&] {
        dumb_add_kernel<scalar_t><<<blocks, threads>>>(
            a_tens.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            b_tens.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dest.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>());
    }));

    return dest;
}

template<typename scalar_t>
__global__ void sparse_time_domain_movie_upsample_kernel(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> movie_frames,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> us_dest,
        const torch::PackedTensorAccessor<int64_t, 2, torch::RestrictPtrTraits, size_t> frame_selection,
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> frame_weights) {

    int64_t h_index = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t h_stride = blockDim.y * gridDim.y;

    int64_t w_index = blockIdx.z * blockDim.z + threadIdx.z;
    int64_t w_stride = blockDim.z * gridDim.z;

    int64_t time_index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t time_stride = blockDim.x * gridDim.x;

    const int64_t height = us_dest.size(0);
    const int64_t width = us_dest.size(1);
    const int64_t max_N = us_dest.size(2);


    for (int64_t t = time_index; t < max_N; t += time_stride) {
        int64_t first_frame_ix = frame_selection[t][0];
        scalar_t first_weight = frame_weights[t][0];

        int64_t second_frame_ix = frame_selection[t][1];
        scalar_t second_weight = frame_weights[t][1];

        for (int64_t h = h_index; h < height; h += h_stride) {
            for (int64_t w = w_index; w < width; w += w_stride) {
                scalar_t first_val = movie_frames[first_frame_ix][h][w];
                scalar_t second_val = movie_frames[second_frame_ix][h][w];

                scalar_t write_val = first_val * first_weight + second_val * second_weight;
                us_dest[t][h][w] = write_val;
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

    torch::Tensor dest = torch::zeros(std::vector<int64_t>({n_bins, height, width}), options);

    const int n_bin_grid = 8;
    const dim3 threads(n_bin_grid, 8, 8);
    const dim3 blocks((n_bin_grid + n_bins - 1) / n_bin_grid, 4, 4);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "sparse_upsample_movie", [&] {
        sparse_time_domain_movie_upsample_kernel<scalar_t><<<blocks, threads>>>(
                movie_frames.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                frame_selection.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
                frame_weights.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}


template<typename scalar_t>
__global__ void sparse_time_domain_upsample_T_kernel(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> movie_frames,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> us_dest,
        const torch::PackedTensorAccessor<int64_t, 2, torch::RestrictPtrTraits, size_t> frame_selection,
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> frame_weights) {

    int64_t h_index = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t h_stride = blockDim.y * gridDim.y;

    int64_t w_index = blockIdx.z * blockDim.z + threadIdx.z;
    int64_t w_stride = blockDim.z * gridDim.z;

    int64_t time_index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t time_stride = blockDim.x * gridDim.x;

    const int64_t height = us_dest.size(0);
    const int64_t width = us_dest.size(1);
    const int64_t max_N = us_dest.size(2);


    for (int64_t t = time_index; t < max_N; t += time_stride) {
        int64_t first_frame_ix = frame_selection[t][0];
        scalar_t first_weight = frame_weights[t][0];

        int64_t second_frame_ix = frame_selection[t][1];
        scalar_t second_weight = frame_weights[t][1];

        for (int64_t h = h_index; h < height; h += h_stride) {
            for (int64_t w = w_index; w < width; w += w_stride) {
                scalar_t first_val = movie_frames[first_frame_ix][h][w];
                scalar_t second_val = movie_frames[second_frame_ix][h][w];

                scalar_t write_val = first_val * first_weight + second_val * second_weight;
                us_dest[h][w][t] = write_val;
            }
        }
    }
}


torch::Tensor upsample_transpose_sparse_movie_cuda(torch::Tensor movie_frames,
                                                   torch::Tensor frame_selection,
                                                   torch::Tensor frame_weights) {

    int64_t n_bins = frame_selection.size(0);
    int64_t height = movie_frames.size(1);
    int64_t width = movie_frames.size(2);

    auto options = torch::TensorOptions()
            .dtype(frame_weights.dtype())
            .layout(torch::kStrided)
            .device(movie_frames.device());

    torch::Tensor dest = torch::zeros(std::vector<int64_t>({height, width, n_bins}), options);

    const int n_bin_grid = 8;
    const dim3 threads(n_bin_grid, 8, 8);
    const dim3 blocks((n_bin_grid + n_bins - 1) / n_bin_grid, 4, 4);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "sparse_upsample_transpose_movie", [&] {
        sparse_time_domain_upsample_T_kernel<scalar_t><<<blocks, threads>>>(
                movie_frames.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                frame_selection.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
                frame_weights.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}


template<typename scalar_t>
__global__ void sparse_time_domain_upsample_flat_kernel(
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> flat_frames,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> us_dest,
        const torch::PackedTensorAccessor<int64_t, 2, torch::RestrictPtrTraits, size_t> frame_selection,
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> frame_weights) {

    const int64_t max_N = us_dest.size(0);
    const int64_t n_pix = us_dest.size(1);

    int64_t t_index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t t_stride = blockDim.x * gridDim.x;

    int64_t p_index = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t p_stride = blockDim.y * gridDim.y;

    for (int64_t t = t_index; t < max_N; t += t_stride) {
        int64_t first_frame_ix = frame_selection[t][0];
        scalar_t first_weight = frame_weights[t][0];

        int64_t second_frame_ix = frame_selection[t][1];
        scalar_t second_weight = frame_weights[t][1];
        for (int64_t p = p_index; p < n_pix; p += p_stride) {

            scalar_t first_val = flat_frames[first_frame_ix][p];
            scalar_t second_val = flat_frames[second_frame_ix][p];

            scalar_t write_val = first_val * first_weight + second_val * second_weight;
            us_dest[t][p] = write_val;
        }
    }
}

torch::Tensor upsample_flat_cuda(torch::Tensor flat_time_data,
                                           torch::Tensor selection,
                                           torch::Tensor weights) {

    int64_t n_bins = selection.size(0);
    int64_t n_pix = flat_time_data.size(1);

    auto options = torch::TensorOptions()
            .dtype(weights.dtype())
            .layout(torch::kStrided)
            .device(flat_time_data.device());

    torch::Tensor dest = torch::zeros(std::vector<int64_t>({n_bins, n_pix}), options);

    const int time_threads = 32;
    const int pix_threads = 16;

    const dim3 threads(time_threads, pix_threads);
    const dim3 blocks((n_bins + time_threads - 1) / time_threads,
                      (n_pix + pix_threads - 1) / pix_threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "sparse_upsample_flat", [&] {
        sparse_time_domain_upsample_flat_kernel<scalar_t><<<blocks, threads>>>(
                flat_time_data.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                selection.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
                weights.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}


template<typename scalar_t>
__global__ void sparse_time_domain_upsample_flat_T_kernel(
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> flat_frames,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> us_dest,
        const torch::PackedTensorAccessor<int64_t, 2, torch::RestrictPtrTraits, size_t> frame_selection,
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> frame_weights) {

    const int64_t n_pix = us_dest.size(0);
    const int64_t max_N = us_dest.size(1);

    int64_t t_index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t t_stride = blockDim.x * gridDim.x;

    int64_t p_index = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t p_stride = blockDim.y * gridDim.y;

    for (int64_t t = t_index; t < max_N; t += t_stride) {
        int64_t first_frame_ix = frame_selection[t][0];
        scalar_t first_weight = frame_weights[t][0];

        int64_t second_frame_ix = frame_selection[t][1];
        scalar_t second_weight = frame_weights[t][1];
        for (int64_t p = p_index; p < n_pix; p += p_stride) {

            scalar_t first_val = flat_frames[first_frame_ix][p];
            scalar_t second_val = flat_frames[second_frame_ix][p];

            scalar_t write_val = first_val * first_weight + second_val * second_weight;
            us_dest[p][t] = write_val;
        }
    }
}


torch::Tensor upsample_transpose_flat_cuda(torch::Tensor flat_time_data,
                                           torch::Tensor selection,
                                           torch::Tensor weights) {
    /*
     *
     * @param flat_time_data
     * @param selection
     * @param weights
     */

    int64_t n_bins = selection.size(0);
    int64_t n_pix = flat_time_data.size(1);

    auto options = torch::TensorOptions()
            .dtype(weights.dtype())
            .layout(torch::kStrided)
            .device(flat_time_data.device());

    torch::Tensor dest = torch::zeros(std::vector<int64_t>({n_pix, n_bins}), options);

    const int time_threads = 32;
    const int pix_threads = 16;

    const dim3 threads(time_threads, pix_threads);
    const dim3 blocks((n_bins + time_threads - 1) / time_threads,
                      (n_pix + pix_threads - 1) / pix_threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "sparse_upsample_transpose_flat", [&] {
        sparse_time_domain_upsample_flat_T_kernel<scalar_t><<<blocks, threads>>>(
                flat_time_data.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                selection.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
                weights.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}
