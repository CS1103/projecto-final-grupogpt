//
// Created by migue on 6/07/2025.
//

#ifndef COMMON_HELPERS_H
#define COMMON_HELPERS_H

#pragma once

#include "../nn/neural_network.h"
#include <iostream>
#include <iomanip>

namespace utec::utils {

    inline int get_predicted_class(const utec::neural_network::Tensor2D<double>& prediction) {
        double max_val = -1.0;
        int max_idx = -1;
        for (size_t i = 0; i < prediction.shape()[1]; ++i) {
            if (prediction(0, i) > max_val) {
                max_val = prediction(0, i);
                max_idx = static_cast<int>(i);
            }
        }
        return max_idx;
    }

    inline void evaluate(utec::neural_network::NeuralNetwork<double>& model,
                         const utec::neural_network::Tensor2D<double>& test_images,
                         const utec::neural_network::Tensor2D<double>& test_labels) {
        int correct_predictions = 0;
        for (size_t i = 0; i < test_images.shape()[0]; ++i) {
            utec::neural_network::Tensor2D<double> image(1, test_images.shape()[1]);
            for(size_t j = 0; j < test_images.shape()[1]; ++j) {
                image(0, j) = test_images(i, j);
            }

            auto logits = model.predict(image);
            // Argmax para predicción
            size_t pred_idx = 0;
            double best = logits(0, 0);
            for (size_t j = 1; j < logits.shape()[1]; ++j)
                if (logits(0, j) > best) { best = logits(0, j); pred_idx = j; }
            // Argmax para etiqueta
            size_t label_idx = 0;
            for (size_t j = 1; j < test_labels.shape()[1]; ++j)
                if (test_labels(i, j) > test_labels(i, label_idx)) label_idx = j;

            if (pred_idx == label_idx) {
                correct_predictions++;
            }
        }
        double accuracy = static_cast<double>(correct_predictions) / test_images.shape()[0];
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy * 100.0 << "%" << std::endl;
    }

}

#endif //COMMON_HELPERS_H
