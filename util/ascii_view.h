#ifndef ASCII_VIEW_H
#define ASCII_VIEW_H
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include "../nn/neural_network.h"

namespace utec::ascii_view {

    template <typename T>
    void print_bw(const Tensor2D<T>& sample) {
        for (int y = 0; y < 28; ++y) {
            for (int x = 0; x < 28; ++x) {
                T v = sample(0, y * 28 + x);     // acceso lineal
                if (v <= 1) v *= (T)255;         // si ya está 0‑1 → 0‑255
                std::cout << (v > 128 ? "#" : " ");
            }
            std::cout << '\n';
        }
    }

    template <typename T>
    void save_pgm(const Tensor2D<T>& sample, const std::string& name) {
        std::ofstream out(name);
        out << "P2\n28 28\n255\n";
        for (int i = 0; i < 784; ++i) {
            T v = sample(0, i);
            if (v <= 1) v *= (T)255;
            out << static_cast<int>(v) << (i % 28 == 27 ? '\n' : ' ');
        }
    }

    template <typename T>
    void inspect(const utec::neural_network::NeuralNetwork<T>& net,
                 const Tensor2D<T>& images,
                 const Tensor2D<T>& labels,
                 size_t idx,
                 bool dump_pgm = false)
    {
        // 1×784 => sample
        Tensor2D<T> sample(1, 784);
        for (size_t k = 0; k < 784; ++k)
            sample(0, k) = images(idx, k);

        // ---- forward & predicción ----
        auto logits = net.forward(sample);
        size_t pred = 0;
        for (size_t j = 1; j < 10; ++j)
            if (logits(0, j) > logits(0, pred)) pred = j;

        // ---- etiqueta real (argmax one‑hot) ----
        size_t label = 0;
        for (size_t j = 1; j < 10; ++j)
            if (labels(idx, j) > labels(idx, label)) label = j;

        std::cout << "\n[Sample " << idx << "]  Pred: " << pred
                  << "  Label: " << label << "\n";
        print_bw(sample);

        if (dump_pgm) {
            std::string fname = "mnist_" + std::to_string(idx) + ".pgm";
            save_pgm(sample, fname);
            std::cout << " >> Guardado " << fname << '\n';
        }
    }
} // namespace utec::ascii_view
#endif //ASCII_VIEW_H
