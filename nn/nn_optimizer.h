#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#pragma once

#include "nn_interfaces.h"

namespace utec::neural_network {

    template <typename T>
    class SGD : public IOptimizer<T> {
    private:
        T _lr;
    public:
        explicit SGD(T learning_rate = 0.01) : _lr(learning_rate) {}
        void update(Tensor<T,2>& params, const Tensor<T,2>& grads) override {
            params = params - (grads * _lr);
        }
    };

    template <typename T>
    class Adam : public IOptimizer<T> {
    private:
        T _lr, _beta1, _beta2, _eps;
    public:
        Adam(T lr = 0.001, T beta1 = 0.9, T beta2 = 0.999, T eps = 1e-8)
            : _lr(lr), _beta1(beta1), _beta2(beta2), _eps(eps) {}

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
            static thread_local Tensor<T, 2> m, v;
            static thread_local int t = 0;
            if (m.shape() != params.shape()) {
                m = Tensor2D<T>(params.shape());
                m.fill(0);
            }
            if (v.shape() != params.shape()) {
                v = Tensor2D<T>(params.shape());
                v.fill(0);
            }
            t++;

            m = m * _beta1 + grads * (1 - _beta1);
            v = v * _beta2 + (grads*grads) * (1 - _beta2);

            auto m_hat = m * (1.0 / (1.0 - std::pow(_beta1, t)));
            auto v_hat = v * (1.0 / (1.0 - std::pow(_beta2, t)));

            for(size_t i=0; i < params.size(); ++i) {
                params[i] -= _lr * m_hat[i] / (std::sqrt(v_hat[i]) + _eps);
            }
        }
    };
}

#endif //NN_OPTIMIZER_H
