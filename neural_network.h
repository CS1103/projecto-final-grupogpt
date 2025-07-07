//
// Created by migue on 6/07/2025.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#pragma once

#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <vector>
#include <memory>

namespace utec::neural_network {

    template <typename T>
    class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> _layers;
    public:
        NeuralNetwork() = default;

        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            _layers.push_back(std::move(layer));
        }

        Tensor2D<T> forward(const Tensor2D<T>& x) {
            Tensor2D<T> current_output = x;
            for (const auto& layer : _layers) {
                current_output = layer->forward(current_output);
            }
            return current_output;
        }

        Tensor2D<T> predict(const Tensor2D<T>& x) {
            return forward(x);
        }

        void backward(const Tensor2D<T>& grad) {
            Tensor2D<T> current_grad = grad;
            for (auto it = _layers.rbegin(); it != _layers.rend(); ++it) {
                current_grad = (*it)->backward(current_grad);
            }
        }

        std::vector<std::unique_ptr<ILayer<T>>>& get_layers() {
            return _layers;
        }
    };
}

#endif //NEURAL_NETWORK_H
