#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#pragma once

#include "nn_interfaces.h"

namespace utec::neural_network {

    template <typename T>
    class ReLU : public ILayer<T> {
    private:
        Tensor2D<T> _mask;
    public:
        ReLU() = default;
        Tensor2D<T> forward(const Tensor2D<T>& x) override {
            _mask = Tensor2D<T>(x.shape());
            Tensor2D<T> output(x.shape());
            for (size_t i = 0; i < x.size(); ++i) {
                output[i] = (x[i] > 0) ? x[i] : 0;
                _mask[i] = (x[i] > 0) ? 1 : 0;
            }
            return output;
        }
        Tensor2D<T> backward(const Tensor2D<T>& grad) override {
            return grad * _mask;
        }
    };

}

#endif //NN_ACTIVATION_H
