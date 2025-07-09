#ifndef TENSOR_H
#define TENSOR_H

#pragma once

#include <vector>
#include <array>
#include <numeric>
#include <stdexcept>
#include <concepts>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <initializer_list>
#include <functional>

namespace utec::algebra {

template <typename T, size_t Rank>
class Tensor {
private:
    std::vector<T> _data;
    std::array<size_t, Rank> _shape;
    std::array<size_t, Rank> _strides;

    void calculate_strides() {
        if constexpr (Rank > 0) {
            _strides[Rank - 1] = 1;
            for (int i = Rank - 2; i >= 0; --i) {
                _strides[i] = _strides[i + 1] * _shape[i + 1];
            }
        }
    }

    template <typename... Dims>
    size_t get_flat_index(size_t first, Dims... rest) const {
        std::array<size_t, Rank> indices = {first, static_cast<size_t>(rest)...};
        size_t flat_index = 0;
        for (size_t i = 0; i < Rank; ++i) {
            flat_index += indices[i] * _strides[i];
        }
        return flat_index;
    }

    // Helper for variadic constructor
    template <typename... Dims>
    static std::array<size_t, Rank> make_shape(Dims... dims) {
        return std::array<size_t, Rank>{static_cast<size_t>(dims)...};
    }

public:
    Tensor() : _shape({}), _strides({}) {}

    Tensor(const std::array<size_t, Rank>& shape) : _shape(shape) {
        size_t total_size = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());
        _data.resize(total_size);
        calculate_strides();
    }

    template <typename... Dims>
    Tensor(Dims... dims) : Tensor(make_shape(dims...)) {
        static_assert(sizeof...(Dims) == Rank, "Number of dimensions must match Rank");
    }

    const std::array<size_t, Rank>& shape() const noexcept { return _shape; }
    size_t size() const noexcept { return _data.size(); }

    auto begin() { return _data.begin(); }
    auto end() { return _data.end(); }
    auto begin() const { return _data.cbegin(); }
    auto end() const { return _data.cend(); }

    template <typename... Idxs>
    T& operator()(Idxs... idxs) {
        static_assert(sizeof...(Idxs) == Rank, "Number of indices must match Rank");
        return _data[get_flat_index(idxs...)];
    }

    template <typename... Idxs>
    const T& operator()(Idxs... idxs) const {
        static_assert(sizeof...(Idxs) == Rank, "Number of indices must match Rank");
        return _data[get_flat_index(idxs...)];
    }

    T& operator[](size_t index) { return _data[index]; }
    const T& operator[](size_t index) const { return _data[index]; }

    void fill(const T& value) noexcept { std::fill(_data.begin(), _data.end(), value); }

    Tensor& operator=(std::initializer_list<T> list) {
        if (list.size() != _data.size()) throw std::invalid_argument("Initializer list size mismatch.");
        std::copy(list.begin(), list.end(), _data.begin());
        return *this;
    }

    Tensor operator+(const Tensor& other) const {
        if constexpr (Rank == 2) {
            if (this->_shape[0] > 1 && other._shape[0] == 1 && this->_shape[1] == other._shape[1]) {
                Tensor result(this->_shape);
                for(size_t i = 0; i < this->_shape[0]; ++i) {
                    for(size_t j = 0; j < this->_shape[1]; ++j) {
                        result(i, j) = (*this)(i, j) + other(0, j);
                    }
                }
                return result;
            }
        }
        if (_shape != other._shape) {
            std::cerr << "[TENSOR ERROR] Shape mismatch in operator+: (";
            for (size_t i = 0; i < Rank; ++i) std::cerr << _shape[i] << (i+1<Rank?", ":"");
            std::cerr << ") vs (";
            for (size_t i = 0; i < Rank; ++i) std::cerr << other._shape[i] << (i+1<Rank?", ":"");
            std::cerr << ")\n";
        }
        assert(_shape == other._shape);
        Tensor result(_shape);
        for(size_t i = 0; i < _data.size(); ++i) result[i] = _data[i] + other[i];
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        assert(_shape == other._shape);
        Tensor result(_shape);
        for(size_t i = 0; i < _data.size(); ++i) result[i] = _data[i] - other[i];
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        assert(_shape == other._shape);
        Tensor result(_shape);
        for(size_t i = 0; i < _data.size(); ++i) result[i] = _data[i] * other[i];
        return result;
    }

    Tensor operator*(const T& scalar) const {
        Tensor result(_shape);
        for(size_t i=0; i<_data.size(); ++i) result[i] = _data[i] * scalar;
        return result;
    }

    Tensor operator/(const T& scalar) const {
        Tensor result(_shape);
        for(size_t i=0; i<_data.size(); ++i) result[i] = _data[i] / scalar;
        return result;
    }

    Tensor<T, 2> transpose_2d() const {
        static_assert(Rank == 2);
        Tensor<T, 2> result(_shape[1], _shape[0]);
        for (size_t i = 0; i < _shape[0]; ++i)
            for (size_t j = 0; j < _shape[1]; ++j)
                result(j, i) = (*this)(i, j);
        return result;
    }

    Tensor<T, 2> matmul(const Tensor<T, 2>& other) const {
        static_assert(Rank == 2);
        assert(_shape[1] == other._shape[0]);
        Tensor<T, 2> result(_shape[0], other._shape[1]);
        for (size_t i = 0; i < _shape[0]; ++i) {
            for (size_t j = 0; j < other._shape[1]; ++j) {
                T sum = 0;
                for (size_t k = 0; k < _shape[1]; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
};

template <typename T, size_t Rank>
std::ostream& operator<<(std::ostream& os, const Tensor<T, Rank>& tensor) {
    if (tensor.size() == 0) {
        os << "{}";
        return os;
    }
    os << "{\n";
    if constexpr (Rank == 2) {
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                os << tensor(i, j) << (j == tensor.shape()[1] - 1 ? "" : " ");
            }
            if (i < tensor.shape()[0] - 1) os << "\n";
        }
    } else {
        for(size_t i = 0; i < tensor.size(); ++i) {
             os << tensor[i] << (i == tensor.size() - 1 ? "" : " ");
        }
    }
    os << "\n}";
    return os;
}
}

#endif //TENSOR_H
