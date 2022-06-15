//
// Created by Eric Wu on 6/15/22.
//
#include <torch/extension.h>

torch::Tensor upsample_sparse_movie_cuda(torch::Tensor movie_frames,
                                         torch::Tensor frame_selection,
                                         torch::Tensor frame_weights) {
    int64_t n_bins = frame_selection.size(0);
    int64_t height = movie_frames.size(1);
    int64_t width = movie_frames.size(2);

    torch::Tensor dest = torch::empty({n_bins, height, width},
                                      torch::dtype(frame_weights.type()).device(movie_frames.device()));

    const int threads = 1024;
    const dim3 blocks((n_bins + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(dest.type(), "sparse_upsample_movie", ([&] {
        sparse_time_domain_movie_upsample_kernel < scalar_t ><<<blocks, threads>>>(
            movie_frames.data<scalar_t>(),
                    dest.data<scalar_t>(),
                    frame_selection.data<scalar_t>(),
                    frame_weights.data<scalar_t>())
    }));

    return dest;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("sparse_upsample_cuda", &upsample_sparse_movie_cuda, "Sparse movie upsampling on GPU");
}

