//
// Created by Eric Wu on 6/15/22.
//
#include <torch/extension.h>

torch::Tensor upsample_sparse_movie_cuda(torch::Tensor movie_frames,
                                         torch::Tensor frame_selection,
                                         torch::Tensor frame_weights);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor movie_sparse_upsample_cuda(torch::Tensor movie_frames,
                                         torch::Tensor frame_selection,
                                         torch::Tensor frame_weights) {
    return upsample_sparse_movie_cuda(movie_frames, frame_selection, frame_weights);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("movie_sparse_upsample_cuda", &movie_sparse_upsample_cuda, "Sparse movie upsampling on GPU");
}
