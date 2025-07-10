#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include "nn/neural_network.h"
#include "utils/tensor.h"
#include "utils/image_processor.h"

using namespace utec::neural_network;
namespace fs = std::filesystem;

void load_layer_weights(NeuralNetwork<double>& net) {
    auto& L = net.get_layers();
    std::cout << "[INFO] Loading weights for each layer..." << std::endl;
    try {
        dynamic_cast<Dense<double>*>(L[0].get())->load_weights("layer0_weights.txt", "layer0_biases.txt");
        std::cout << "[INFO] Loaded layer0 weights and biases." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load layer0 weights: " << e.what() << std::endl;
    }
    try {
        dynamic_cast<Dense<double>*>(L[2].get())->load_weights("layer2_weights.txt", "layer2_biases.txt");
        std::cout << "[INFO] Loaded layer2 weights and biases." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load layer2 weights: " << e.what() << std::endl;
    }
    try {
        dynamic_cast<Dense<double>*>(L[4].get())->load_weights("layer4_weights.txt", "layer4_biases.txt");
        std::cout << "[INFO] Loaded layer4 weights and biases." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load layer4 weights: " << e.what() << std::endl;
    }
}

int main() {
    try {
        // Cargar imagen y procesar
        auto input = utec::utils::preprocess_image_stb("../Imagenes_Prueba/ceronegro.png");



        // Cargar red neuronal
        NeuralNetwork<double> net;
        net.add_layer(std::make_unique<Dense<double>>(784, 128));
        net.add_layer(std::make_unique<ReLU<double>>());
        net.add_layer(std::make_unique<Dense<double>>(128, 64));
        net.add_layer(std::make_unique<ReLU<double>>());
        net.add_layer(std::make_unique<Dense<double>>(64, 10));

        load_layer_weights(net);

        // Predecir
        auto output = net.forward(input);
        size_t pred = 0;
        for (size_t i = 1; i < 10; ++i) {
            if (output(0, i) > output(0, pred))
                pred = i;
        }

        std::cout << "Predicción: " << pred << std::endl;

        // Mostrar representación ASCII
        std::cout << "\nASCII representation:\n";
        utec::utils::print_ascii_28x28(input);
        std::cout << "\n[Final 28x28 normalized view]\n";
        utec::utils::print_ascii_28x28(input);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
