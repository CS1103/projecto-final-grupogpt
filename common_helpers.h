//
// Created by migue on 6/07/2025.
//

#ifndef COMMON_HELPERS_H
#define COMMON_HELPERS_H

#pragma once

#include "neural_network.h"
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
            utec::neural_network::Tensor2D<double> image(1, 28 * 28);
            for(size_t j = 0; j < 28*28; ++j) {
                image(0, j) = test_images(i, j);
            }

            auto logits = model.predict(image);
            int predicted_class = get_predicted_class(logits);

            int true_class = -1;
            for(size_t j = 0; j < test_labels.shape()[1]; ++j) {
                if(test_labels(i, j) == 1.0) {
                    true_class = static_cast<int>(j);
                    break;
                }
            }

            if (predicted_class == true_class) {
                correct_predictions++;
            }
        }
        double accuracy = static_cast<double>(correct_predictions) / test_images.shape()[0];
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy * 100.0 << "%" << std::endl;
    }

}

#endif //COMMON_HELPERS_H
