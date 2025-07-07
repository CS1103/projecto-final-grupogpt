//
// Created by migue on 6/07/2025.
//

#ifndef NN_LOSS_H
#define NN_LOSS_H

#pragma once

#include "nn_interfaces.h"

namespace utec::neural_network {

    // Softmax + CrossEntropy Loss for multi-class classification
    template<typename T>
    class SoftmaxCrossEntropyLoss {
    private:
        Tensor2D<T> _softmax_outputs;
        Tensor2D<T> _last_targets;
    public:
        SoftmaxCrossEntropyLoss() = default;

        T forward(const Tensor2D<T>& logits, const Tensor2D<T>& targets) {
            _last_targets = targets;
            _softmax_outputs = Tensor2D<T>(logits.shape());
            T total_loss = 0;

            for (size_t i = 0; i < logits.shape()[0]; ++i) {
                T max_logit = logits(i, 0);
                for (size_t j = 1; j < logits.shape()[1]; ++j) {
                    if (logits(i, j) > max_logit) max_logit = logits(i, j);
                }

                T sum_exp = 0;
                for (size_t j = 0; j < logits.shape()[1]; ++j) {
                    T exp_val = std::exp(logits(i, j) - max_logit);
                    _softmax_outputs(i, j) = exp_val;
                    sum_exp += exp_val;
                }

                for (size_t j = 0; j < logits.shape()[1]; ++j) {
                    _softmax_outputs(i, j) /= sum_exp;
                    if (targets(i, j) == 1) {
                        total_loss += -std::log(_softmax_outputs(i, j) + 1e-9);
                    }
                }
            }
            return total_loss / logits.shape()[0];
        }

        Tensor2D<T> backward() {
            return (_softmax_outputs - _last_targets) / static_cast<T>(_last_targets.shape()[0]);
        }
    };
}

#endif //NN_LOSS_H
