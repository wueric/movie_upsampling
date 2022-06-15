#define INVALID_IDX -1

template <typename scalar_t>
__global__ void sparse_time_domain_movie_upsample_kernel(
        const torch::PackedTensorAccessor64<scalar_t,3,torch::RestrictPtrTraits> movie_frames,
        torch::PackedTensorAccessor64<scalar_t,3,torch::RestrictPtrTraits> us_dest,
        const torch::PackedTensorAccessor64<int64_t,2,torch::RestrictPtrTraits> frame_selection,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> frame_weights) {

    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;

    const int64_t max_N = frame_selection.size(0);
    const int64_t height = frame_selection.size(1);
    const int64_t width = frame_selection.size(2);

    for (int64_t i = index; i < max_n; i += stride) {
        if (frame_selection[i][1]; == INVALID_IDX) {
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
