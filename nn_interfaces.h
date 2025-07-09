//
// Created by migue on 6/07/2025.
//

#ifndef NN_INTERFACES_H
#define NN_INTERFACES_H

#pragma once

#include "tensor.h"

namespace utec::neural_network {

    using utec::algebra::Tensor;

    template<typename T>
    using Tensor2D = utec::algebra::Tensor<T, 2>;

    template<typename T>
    struct ILayer {
        virtual ~ILayer() = default;
        virtual Tensor2D<T> forward(const Tensor2D<T>& x) = 0;
        virtual Tensor2D<T> backward(const Tensor2D<T>& grad) = 0;
        virtual void update(T learning_rate) {}
    };
}

#endif //NN_INTERFACES_H
