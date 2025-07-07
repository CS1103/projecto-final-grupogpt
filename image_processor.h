//
// Created by migue on 6/07/2025.
//

#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include "nn_interfaces.h"

namespace utec::utils {

// Processes an image from a file path into a tensor suitable for the MNIST model.
utec::neural_network::Tensor2D<double> process_image_for_mnist(const std::string& file_path) {
    int width, height, channels;

    // Load the image from the file path provided by the user
    unsigned char* img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        throw std::runtime_error("Failed to load image: " + file_path + ". Check the path and file integrity.");
    }

    // --- Image Processing Pipeline ---
    const int mnist_width = 28;
    const int mnist_height = 28;
    const int mnist_size = mnist_width * mnist_height;

    // 1. Resize the image to 28x28 pixels
    std::vector<unsigned char> resized_data(mnist_size * channels);
    int success = stbir_resize_uint8(img_data, width, height, 0,
                                     resized_data.data(), mnist_width, mnist_height, 0,
                                     channels);
    stbi_image_free(img_data); // Free the original, large image data immediately

    if (!success) {
        throw std::runtime_error("Failed to resize image.");
    }

    // 2. Convert to grayscale, invert colors, and normalize into the final Tensor
    utec::neural_network::Tensor2D<double> result(1, mnist_size);
    for (int i = 0; i < mnist_size; ++i) {
        double gray_value = 0.0;

        // Convert to grayscale if the image has color channels (RGB/RGBA)
        if (channels >= 3) {
            // Standard luminance formula to convert RGB to Grayscale
            gray_value = 0.299 * resized_data[i * channels] +
                         0.587 * resized_data[i * channels + 1] +
                         0.114 * resized_data[i * channels + 2];
        } else { // Already grayscale
            gray_value = resized_data[i * channels];
        }

        // Invert colors: MNIST uses white digits (255) on a black background (0).
        // A user will typically draw a black digit (0) on a white background (255).
        // This step makes the user's image look like the training data.
        double inverted_value = 255.0 - gray_value;

        // Normalize the pixel value to the [0.0, 1.0] range
        result(0, i) = inverted_value / 255.0;
    }

    return result;
}

}

#endif //IMAGE_PROCESSOR_H
