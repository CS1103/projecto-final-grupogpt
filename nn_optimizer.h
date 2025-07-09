//
// Created by migue on 6/07/2025.
//

#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#pragma once

#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

    template <typename T>
    class Adam {
    private:
        T _lr;
        T _beta1, _beta2, _eps;
        Tensor2D<T> _m, _v;
        int _t = 0;

    public:
        Adam(T lr = 0.001, T beta1 = 0.9, T beta2 = 0.999, T eps = 1e-8)
            : _lr(lr), _beta1(beta1), _beta2(beta2), _eps(eps) {}

        void set_learning_rate(T lr) { _lr = lr; }

        void update(Tensor2D<T>& params, const Tensor2D<T>& grads) {
            if (_t == 0) {
                _m = Tensor2D<T>(params.shape());
                _v = Tensor2D<T>(params.shape());
                _m.fill(0);
                _v.fill(0);
            }
            _t++;

            _m = (_m * _beta1) + (grads * (1.0 - _beta1));
            _v = (_v * _beta2) + ((grads * grads) * (1.0 - _beta2));

            Tensor2D<T> m_hat = _m / (1.0 - std::pow(_beta1, _t));
            Tensor2D<T> v_hat = _v / (1.0 - std::pow(_beta2, _t));

            for(size_t i = 0; i < params.size(); ++i) {
                params[i] -= _lr * m_hat[i] / (std::sqrt(v_hat[i]) + _eps);
            }
        }
    };
}

#endif //NN_OPTIMIZER_H
