#define INVALID_IDX -1
#define FIRST_OVERLAP 0
#define SECOND_OVERLAP 1

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template<typename scalar_t>
__global__ void _cu_time_upsample_forward(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> flat_noupsample,
        const torch::PackedTensorAccessor<int64_t, 3, torch::RestrictPtrTraits, size_t> flat_selection,
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> flat_weights,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> flat_upsample) {

    const int64_t nframes_noupsample = flat_noupsample.size(1);
    int64_t f_index = threadIdx.y + blockIdx.y * blockDim.y;
    int64_t f_stride = blockDim.y * gridDim.y;

    const int64_t n_pix = flat_noupsample.size(2);
    int64_t p_index = threadIdx.z + blockIdx.z * blockDim.z;
    int64_t p_stride = blockDim.z * gridDim.z;

    int64_t b = blockIdx.x;

    for (int64_t f = f_index; f < nframes_noupsample, f+= f_stride) {
        if (flat_selection[b][f][SECOND_OVERLAP] == INVALID_IDX) {
            int64_t only_frame_ix = flat_selection[b][f][FIRST_OVERLAP];

            for (int64_t p = p_index; p < n_pix; p += p_stride) {
                flat_upsample[b][f][p]  = flat_noupsample[b][only_frame_ix][p];
            }
        } else {

            int64_t first_frame_ix = flat_selection[b][f][FIRST_OVERLAP];
            int64_t second_frame_ix = flat_selection[b][f][SECOND_OVERLAP];

            scalar_t first_frame_w = flat_weights[b][f][FIRST_OVERLAP];
            scalar_t second_frame_w = flat_weights[b][f][SECOND_OVERLAP];

            for (int64_t p = p_index; p < n_pix; p += p_stride) {

                scalar_t first_val = flat_noupsample[b][first_frame_ix][p];
                scalar_t second_val = flat_noupsample[b][second_frame_ix][p];

                scalar_t write_val = first_val * first_frame_w + second_val * second_frame_w;
                flat_upsample[b][f][p] = write_val;
            }
        }
    }
}


torch::Tensor _upsample_flat_forward(torch::Tensor flat_noupsample,
                                    torch::Tensor flat_selection,
                                    torch::Tensor flat_weights) {
    /*
     * This function is meant to do the forward pass time upsample during jittered natural
     * image reconstruction. Since this function is targeted for reconstruction only we make
     * the following assumptions about the input sizes.
     *
     *  (1)  batch for frames_noupsample is relatively small, i.e. no more than 16 or so
     *  (2)  n_frames_upsample, the number of frames in the desired upsampled movie is also small,
     *      i.e. no more than 1000 or so
     *  (3) Most of the CUDA performance gains here will be achieved by parallelizing over the multiple
     *      datapoints that occur during the same time window
     *
     * Reason to implement this as a matrix multiplication: if we implement this as a matrix multiplication
     *  we then have to first compute the matrix that we multiply with, which is just as much work
     *  as writing/running this function
     *
     * @param flat_noupsample: shape (batch, n_frames_noupsample, n_pix)
     * @param flat_selection: shape (batch, n_frames_upsample, 2), int64_t
     * @param flat_weights: shape (batch, n_frames_upsample, 2)
     */

    const int64_t batch = flat_noupsample.size(0);
    const int64_t nframes_noupsample = flat_noupsample.size(1);
    const int64_t n_pix = flat_noupsample.size(2);

    const int64_t nframes_upsample = flat_selection.size(1);

    auto options = torch::TensorOptions()
            .dtype(a_tens.dtype())
            .layout(torch::kStrided)
            .device(a_tens.device());
    torch::Tensor dest = torch.zeros(std::vector<int64_t>({batch, nframes_upsample, n_pix}), options);

    const int64_t threads_per_time = 4;
    const dim3 threads(1, threads_per_time, 256);
    const dim3 blocks(batch, (n_frames_upsample + threads_per_time - 1) / threads_per_time, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "_cu_time_upsample_forward", [&] {
        _cu_time_upsample_forward<scalar_t><<<blocks, threads>>>(
                frames_noupsample.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                frame_selection.packed_accessor<int64_t, 3, torch::RestrictPtrTraits, size_t>(),
                frame_weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}


template<typename scalar_t>
__global__ void _cu_time_upsample_backward(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dloss_dupsample,
        const torch::PackedTensorAccessor<int64_t, 3, torch::RestrictPtrTraits, size_t> backward_selection,
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> backward_weights,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dloss_dnoupsample) {

    int64_t b = blockIdx.x;

    const int64_t n_frames_upsample = dloss_dupsample.size(1);
    const int64_t n_frames_noupsample = backward_selection.size(1);
    int64_t f_index = threadIdx.y + blockIdx.y * blockDim.y;
    int64_t f_stride = blockDim.y * gridDim.y;

    const int64_t n_pix = dloss_dupsample.size(2);
    int64_t p_index = threadIdx.z + blockIdx.z * blockDim.z;
    int64_t p_stride = blockDim.z * gridDim.z;

    const int64_t max_overlap_frames = backward_selection.size(2);
    for (int64_t f = f_index; f < n_frames_noupsample; f += f_stride) {
        for (int64_t p = p_index; p < n_pix; p += p_stride) {

            scalar_t acc = 0.0;
            for (int64_t ix = 0; ix < max_overlap_frames; ++ix){

                int64_t read_from_ix = backward_selection[b][f][ix];
                if (read_from_ix == INVALID_IDX) break;

                acc += (dloss_dupsample[b][read_from_ix][p] * backward_weights[b][f][ix]);
            }
            dloss_dnoupsample[b][f][p] = acc;
        }
    }
}


torch::Tensor _upsample_flat_backward(torch::Tensor dloss_dflat_upsample,
                                     torch::Tensor backward_selection,
                                     torch::Tensor backward_weights) {

    /*
     * This is trickier to do parallel backward with, since in order
     * to avoid race conditions, we have to make sure that each entry in
     * the destination is written to by exactly one thread. The extra complexity
     * is because the frame rate may be unstable and is also a non-integer multiple
     * of the bin rate and so there might be a different number of time bins associated
     * with each of frames
     *
     * Constraints: in order to put this into the Pytorch autograd framework, we have
     * to fit this implementation into the torch.autograd.Function forward and backward methods
     * Therefore, we parallelize over batch, n_frames_noupsample, n_pix dimensions
     *
     *
     * @param dloss_dflat_upsample: shape (batch, n_frames_upsample, n_pix)
     * @param backward_selection: shape (batch, n_frames_noupsample, n_max_overlap)
     * @param backward_weights: shape (batch, n_frames_noupsample, n_max_overlap)
     */

    const int64_t batch = dloss_dflat_upsample.size(0);
    const int64_t nframes_noupsample = backward_selection.size(1);
    const int64_t n_pix = dloss_dflat_upsample.size(2);

    auto options = torch::TensorOptions()
            .dtype(a_tens.dtype())
            .layout(torch::kStrided)
            .device(a_tens.device());
    torch::Tensor dest = torch.zeros(std::vector<int64_t>({batch, n_frames_noupsample, n_pix}), options);

    const int64_t threads_per_time = 4;
    const dim3 threads(1, threads_per_time, 256);
    const dim3 blocks(batch, (nframes_noupsample + threads_per_time - 1) / threads_per_time, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dest.scalar_type(), "_cu_time_upsample_backward", [&] {
        _cu_time_upsample_backward<scalar_t><<<blocks, threads>>>(
                dloss_dflat_upsample.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                backward_selection.packed_accessor<int64_t, 3, torch::RestrictPtrTraits, size_t>(),
                backward_weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                dest.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>());
    });

    return dest;
}
