//
// Created by Eric Wu on 6/13/22.
//

#ifndef MOVIE_UPSAMPLING_NDARRAYWRAPPER_H
#define MOVIE_UPSAMPLING_NDARRAYWRAPPER_H

#include <vector>
#include <array>
#include <stdexcept>
#include <string>
#include <sstream>

/*
 * Wrappers for easy read-write access and slicing for C-ordered
 *      pybind11 np.ndarray buffers
 *
 * Implicitly assumes that the underlying buffer from numpy is
 *      C-ordered.
 *
 * DOES NOT SUPPORT NEGATIVE INDEXING!!!
 *
 * DOES NOT ALLOW RESHAPES
 *
 * Supports two distinct modes of access. Both types of wrappers will share
 * a common basic interface.
 * (1) Thin wrapper for direct access to blocks memory, where we store only
 *      a base pointer and maximum dimensions. This allows the user to claim
 *      that a subarray of the np.ndarray has known contiguous dimensions, and
 *      to directly access it. USEFUL WHEN CONSECUTIVE MEMORY ACCESSES ARE LIKELY
 *      AND HIGH SPEED IS NEEDED.
 *
 *      All instance variables in this wrapper will be public, since the goal
 *      is speed and easy implementation. Safety is the users' problem here
 *
 *      We will only support 1D, 2D, and 3D subarrays for this method
 *
 * (2) Wrapper around slices of the np.ndarray, where we store a base pointer, dimensions,
 *      and strides. This allows the user to easily manipulate subarrays of the full
 *      np.ndarray, but is slower because it requires more pointer arithmetic.
 */

namespace CNDArrayWrapper {

    class Slice {

    public:
        static const int64_t LOW = 0;

        static const int64_t MAX = -1;
        static const int64_t ALL = -1;

        virtual int64_t compute_first_idx() const = 0;

        virtual int64_t compute_ceil_idx(int64_t array_dim) const = 0;

        virtual int64_t compute_step() const = 0;

        virtual bool is_singledim() const = 0;

        int64_t compute_dim(int64_t array_dim) const {
            int64_t start = compute_first_idx();
            int64_t end = compute_ceil_idx(array_dim);
            int64_t step = compute_step();

            return (end - start) / step;
        };
    };

    class IdxSlice : public Slice {

    private:
        const int64_t _idx;

    public:
        IdxSlice(int64_t idx) : _idx(idx) {};

        int64_t compute_first_idx() const { return _idx; };

        int64_t compute_ceil_idx(int64_t array_dim) const { return _idx + 1; };

        int64_t compute_step() const { return 0; };

        bool is_singledim() const { return true; };
    };

    class RangeSlice : public Slice {
    private:
        const int64_t _low;
        const int64_t _high;
        const int64_t _step;

    public:
        RangeSlice(int64_t low, int64_t high, int64_t step) : _low(low), _high(high), _step(step) {};

        RangeSlice(int64_t low, int64_t high) : _low(low), _high(high), _step(1) {};

        int64_t compute_first_idx() const { return _low; };

        int64_t compute_ceil_idx(int64_t array_dim) const { return _high; };

        int64_t compute_step() const { return _step; };

        bool is_singledim() const { return false; };
    };

    class InferredSlice : public Slice {
    private:
        const int64_t _low;
        const int64_t _step;

    public:
        InferredSlice() : _low(0), _step(1) {};

        InferredSlice(int64_t low) : _low(low), _step(1) {};

        InferredSlice(int64_t low, int64_t step) : _low(low), _step(step) {};

        int64_t compute_first_idx() const { return _low; };

        int64_t compute_ceil_idx(int64_t array_dim) const {
            int64_t n_steps = (array_dim - _low) / _step;
            return _step * n_steps + _low;
        }

        int64_t compute_step() const { return _step; };

        bool is_singledim() const { return false; };
    };

    std::shared_ptr<Slice> makeRangeSlice(int64_t low, int64_t high, int64_t step) {
        return std::make_shared<RangeSlice>(low, high, step);
    };

    std::shared_ptr<Slice> makeRangeSlice(int64_t low, int64_t high) {
        return std::make_shared<RangeSlice>(low, high);
    };

    std::shared_ptr<Slice> makeIdxSlice(int64_t idx) {
        return std::make_shared<IdxSlice>(idx);
    };

    std::shared_ptr<Slice> makeInferredSlice(int64_t low, int64_t step) {
        return std::make_shared<InferredSlice>(low, step);
    };

    std::shared_ptr<Slice> makeInferredSlice(int64_t low) {
        return std::make_shared<InferredSlice>(low);
    };

    std::shared_ptr<Slice> makeAllSlice() {
        return std::make_shared<InferredSlice>();
    };

    template<uint8_t N>
    std::array<int64_t, N> _auto_stride(std::array<int64_t, N> dim) {
        auto x = std::array<int64_t, N>{};
        int64_t stride = 1;
        for (int64_t i = N - 1; i >= 0; --i) {
            x[i] = stride;
            stride = stride * dim[i];
        }
        return x;
    }


    template<typename T, uint8_t N>
    class StaticNDArrayWrapper {

    public:
        T *const base_ptr;
        const std::array<int64_t, N> shape;
        const std::array<int64_t, N> stride;

        StaticNDArrayWrapper<T, N>(
                T *_base_ptr,
                std::array<int64_t, N> _shape,
                std::array<int64_t, N> _stride) : base_ptr(_base_ptr), shape(_shape), stride(_stride) {};

        StaticNDArrayWrapper<T, N>(
                T *_base_ptr,
                std::array<int64_t, N> _shape) : base_ptr(_base_ptr), shape(_shape),
                                                 stride(_auto_stride<N>(_shape)) {};

        std::string generate_shape_str() const {

            std::ostringstream stream;
            stream << "(";

            for (uint8_t i = 0; i < N - 1; ++i) {
                stream << shape[i] << ", ";
            }

            stream << shape[N - 1] << ")";
            return stream.str();
        }


        template<typename... Ix>
        T *addressIx(Ix... ix) const { return base_ptr + computeOffset(ix...); };

        template<typename... Ix>
        int64_t computeOffset(Ix... ix) const {
            static_assert(sizeof...(ix) == N, "N-dim StaticNDArrayWrapper must be indexed by N indices");

            auto indices = std::array<int64_t, N>{{ix...}};

            int64_t offset = 0;
            for (size_t i = 0; i < N; ++i) {
                if (indices[i] >= shape[i] || indices[i] < 0) {
                    //throw std::out_of_range("index {} out-of-range for dim {}, max {}", indices[i], i, shape[i]);
                    //
                    std::ostringstream stream;
                    stream << "index " << indices[i] << " out-of-range for dim " << i << "; shape "
                           << generate_shape_str();
                    std::string error_string = stream.str();
                    throw std::out_of_range(error_string);
                }
                offset += (stride[i] * indices[i]);
            }

            return offset;
        };

        template<typename... Ix>
        void storeTo(T val, Ix... ix) const { *(addressIx(ix...)) = val; };

        template<typename... Ix>
        T valueAt(Ix... ix) const { return *(addressIx(ix...)); };

        template<uint8_t M, typename... Sl>
        StaticNDArrayWrapper<T, M> slice(Sl... sl) {
            static_assert(sizeof...(sl) == N, "N-dim StaticNDArrayWrapper must be sliced by N indices");

            // first need to compute the base pointer position
            // which is the position of the first thing in the slice
            T *new_base_ptr = addressIx(sl->compute_first_idx()...);

            // then need to compute the new shapes and strides

            auto slices = std::array<std::shared_ptr<Slice>, N>{{sl...}};
            std::array<int64_t, M> output_shapes{};
            std::array<int64_t, M> output_strides{};

            // Logic for figuring out the output shape and stride
            //  Loop over the Slice objects in reverse order (from last index to first index)
            //  For each Slice object, the slice either corresponds to multiple rows, or a single row
            //      If it is a single row, then it does not need an entry in output_shapes or output_strides
            //      If it corresponds to multiple rows, then we need to determine the number of rows and the
            //          step size between the rows
            int64_t output_ax = M - 1;
            for (int64_t ax = N - 1; ax >= 0; --ax) {
                auto current_slice = slices[ax];
                if (!current_slice->is_singledim()) {
                    output_shapes[output_ax] = current_slice->compute_dim(shape[ax]);
                    output_strides[output_ax] = current_slice->compute_step() * stride[ax];
                    --output_ax;
                }
            }

            return StaticNDArrayWrapper<T, M>(new_base_ptr, output_shapes, output_strides);
        }
    };

    template<typename T>
    class OneDRawArrayWrapper {
    public:
        T *array_ptr;
        int64_t dim0;

        OneDRawArrayWrapper<T>(T *ptr, const int64_t dim0_) : array_ptr(ptr), dim0(dim0_) {};

        T valueAt(int64_t ix0) { return *(array_ptr + ix0); }

        void storeTo(T val, int64_t ix0) { *(array_ptr + ix0) = val; }

        T *addressIx(int64_t ix0) { return array_ptr + ix0; }
    };


    template<typename T>
    struct TwoDRawArrayWrapper {

        T *array_ptr;
        int64_t dim0;
        int64_t dim1;

        TwoDRawArrayWrapper<T>(T *ptr, const int64_t dim0_, const int64_t dim1_) : array_ptr(ptr), dim0(dim0_),
                                                                                   dim1(dim1_) {};

        inline T valueAt(int64_t ix0, int64_t ix1) {
            return *(addressIx(ix0, ix1));
        }

        inline void storeTo(T val, int64_t ix0, int64_t ix1) {
            *(addressIx(ix0, ix1)) = val;
        }

        inline T *addressIx(int64_t ix0, int64_t ix1) {
            return array_ptr + ix0 * dim1 + ix1;
        }
    };

    template<typename T>
    struct ThreeDRawArrayWrapper {
        T *array_ptr;
        const int64_t dim0;
        const int64_t dim1;
        const int64_t dim2;

        ThreeDRawArrayWrapper<T>(T *ptr, const int64_t dim0_, const int64_t dim1_, const int64_t dim2_)
                : array_ptr(ptr), dim0(dim0_), dim1(dim1_), dim2(dim2_) {};

        inline T valueAt(int64_t ix0, int64_t ix1, int64_t ix2) {
            return *(addressIx(ix0, ix1, ix2));
        }

        inline void storeTo(T val, int64_t ix0, int64_t ix1, int64_t ix2) {
            *(addressIx(ix0, ix1, ix2)) = val;
        }

        inline T *addressIx(int64_t ix0, int64_t ix1, int64_t ix2) {
            return array_ptr + ix0 * dim1 * dim2 + ix1 * dim2 + ix2;
        }

    };

    template<typename T, size_t N>
    struct NDRawArrayWrapper {
        T *array_ptr;
        const std::array<int64_t, N> shape;
        std::array<int64_t, N> stride;

        NDRawArrayWrapper<T, N>(T *ptr, std::array<int64_t, N> _shape) : array_ptr(ptr), shape(_shape) {

            int64_t acc_stride = 1;
            for (int64_t i = N - 1; i >= 0; --i) {
                stride[i] = acc_stride;
                acc_stride = acc_stride * _shape[i];
            }
        };

        template<typename... Ix>
        inline T valueAt(Ix... ix) {
            return *(addressIx(ix...));
        }

        template<typename... Ix>
        inline void storeTo(T val, Ix... ix) {
            *(addressIx(ix...)) = val;
        }

        template<typename... Ix>
        T *addressIx(Ix... ix) const {
            static_assert(sizeof...(ix) == N, "N-dim NDRawArrayWrapper must be indexed by N indices");
            auto indices = std::array<int64_t, N>{{ix...}};

            int64_t offset = 0;
            for (size_t i = 0; i < N; ++i) {
                offset += stride[i] * indices[i];
            }
            return array_ptr + offset;
        }

    };
}


#endif //MOVIE_UPSAMPLING_NDARRAYWRAPPER_H
