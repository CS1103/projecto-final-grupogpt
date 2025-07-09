#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include "../nn/nn_interfaces.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stb_image_resize.h"

namespace utec::utils {

using Tensor2D = utec::neural_network::Tensor2D<double>;

Tensor2D preprocess_image_stb(const std::string& filepath) {
    int w, h, channels;
    unsigned char* img = stbi_load(filepath.c_str(), &w, &h, &channels, 0);
    if (!img) throw std::runtime_error("Failed to load image: " + filepath);

    std::cout << "[INFO] Image loaded: " << w << "x" << h << ", channels: " << channels << "\n";

    // ── Convert to grayscale ──
    std::vector<unsigned char> gray(w * h);
    for (int i = 0; i < w * h; ++i) {
        int r = img[i * channels + 0];
        int g = (channels > 1) ? img[i * channels + 1] : r;
        int b = (channels > 2) ? img[i * channels + 2] : r;
        gray[i] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
    }
    stbi_image_free(img);


    // ── Binarización ──
    int thr = 60;
    std::vector<unsigned char> mask(w * h, 0);
    for (int i = 0; i < w * h; ++i)
        mask[i] = gray[i] > thr ? 255 : 0;

    // ── Dilatación (1 paso) ──
    std::vector<unsigned char> dil(mask);
    for (int y = 1; y < h - 1; ++y)
        for (int x = 1; x < w - 1; ++x)
            if (mask[y * w + x])
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        dil[(y + dy) * w + (x + dx)] = 255;

    // ── Bounding box usando dil ──
    int top = h, bottom = 0, left = w, right = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (dil[y * w + x]) {
                top = std::min(top, y);
                bottom = std::max(bottom, y);
                left = std::min(left, x);
                right = std::max(right, x);
            }
        }
    if (top >= bottom || left >= right)
        throw std::runtime_error("No se encontró un dígito visible en la imagen.");

    int bw = right - left + 1;
    int bh = bottom - top + 1;
    std::vector<unsigned char> cropped(bw * bh);
    for (int y = 0; y < bh; ++y)
        for (int x = 0; x < bw; ++x)
            cropped[y * bw + x] = gray[(top + y) * w + (left + x)];

    // -- resize manteniendo aspecto --
    int maxSide = 20;
    int new_w, new_h;
    if (bw >= bh) { new_w = maxSide; new_h = bh * maxSide / bw; }
    else          { new_h = maxSide; new_w = bw * maxSide / bh; }
    std::vector<unsigned char> resized(new_w * new_h);
    stbir_resize_uint8(cropped.data(), bw, bh, 0, resized.data(), new_w, new_h, 0, 1);

    // -- centrar en 28x28 --
    std::vector<unsigned char> final(28 * 28, 0);
    int off_x = (28 - new_w) / 2;
    int off_y = (28 - new_h) / 2;
    for (int y = 0; y < new_h; ++y)
        for (int x = 0; x < new_w; ++x)
            final[(y + off_y) * 28 + (x + off_x)] = resized[y * new_w + x];

    // (Opcional) ASCII preview antes de normalizar
    std::cout << "\n[28x28 después de centrar, antes de normalizar]\n";
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            std::cout << (final[y * 28 + x] > 40 ? '#' : ' ');
        }
        std::cout << "\n";
    }

    // ── Convert to normalized Tensor ──
    Tensor2D result(1, 784);
    for (int i = 0; i < 784; ++i)
    {
        double norm = final[i] / 255.0; // fondo negro (0), dígito blanco (1)
        result(0, i) = norm;
    }
    return result;
}

// ── ASCII Preview ──
inline void print_ascii_28x28(const Tensor2D& t)
{
    std::cout << "\n[Final 28x28 normalized view]\n";
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            double v = t(0, y*28 + x);
            std::cout << (v > 0.25 ? "#" : " "); //  ahora ‘#’ será el trazo, fondo en blanco
        }
        std::cout << '\n';
    }
}

} // namespace utec::utils

#endif // IMAGE_PROCESSOR_H
