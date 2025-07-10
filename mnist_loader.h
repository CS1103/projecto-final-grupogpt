//
// Created by migue on 6/07/2025.
//

#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <tuple>
#include "nn/nn_interfaces.h"

namespace utec::data {

    using Tensor2D_d = utec::neural_network::Tensor2D<double>;

    std::vector<std::string> split(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

    std::pair<Tensor2D_d, Tensor2D_d> load_mnist_csv(const std::string& file_path, size_t num_images_to_load) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + file_path);
        }

        const size_t image_size = 28 * 28;
        const size_t num_classes = 10;
        Tensor2D_d images(num_images_to_load, image_size);
        Tensor2D_d labels(num_images_to_load, num_classes);
        labels.fill(0);

        std::string line;
        std::getline(file, line); // Skip header

        size_t current_image = 0;
        while (std::getline(file, line) && current_image < num_images_to_load) {
            auto tokens = split(line, ',');
            int label_val = std::stoi(tokens[0]);
            labels(current_image, label_val) = 1.0;

            for (size_t i = 0; i < image_size; ++i) {
                images(current_image, i) = std::stod(tokens[i + 1]) / 255.0;
            }
            current_image++;
        }

        return {images, labels};
    }
}

#endif //MNIST_LOADER_H
