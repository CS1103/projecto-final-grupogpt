//
// Created by migue on 6/07/2025.
//

#include <iostream>
#include <string>
#include <iomanip>
#include "neural_network.h"
#include "image_processor.h"
#include "common_helpers.h"

using namespace utec::neural_network;

int main() {
    try {
        const size_t IMAGE_SIZE = 784;
        const size_t NUM_CLASSES = 10;

        NeuralNetwork<double> model;
        model.add_layer(std::make_unique<Dense<double>>(IMAGE_SIZE, 128));
        model.add_layer(std::make_unique<ReLU<double>>());
        model.add_layer(std::make_unique<Dense<double>>(128, 64));
        model.add_layer(std::make_unique<ReLU<double>>());
        model.add_layer(std::make_unique<Dense<double>>(64, NUM_CLASSES));

        std::cout << "Loading pre-trained model weights..." << std::endl;
        dynamic_cast<Dense<double>*>(model.get_layers()[0].get())->load_weights("layer0_weights.txt", "layer0_biases.txt");
        dynamic_cast<Dense<double>*>(model.get_layers()[2].get())->load_weights("layer2_weights.txt", "layer2_biases.txt");
        dynamic_cast<Dense<double>*>(model.get_layers()[4].get())->load_weights("layer4_weights.txt", "layer4_biases.txt");

        std::cout << "\n--- Digit Recognizer Ready ---" << std::endl;
        std::string user_input_path;
        while (true) {
            std::cout << "Enter the path to your digit image or type 'exit': ";
            std::getline(std::cin, user_input_path);

            if (user_input_path == "exit") break;

            try {
                Tensor2D<double> image_tensor = utec::utils::process_image_for_mnist(user_input_path);

                Tensor2D<double> logits = model.predict(image_tensor);
                int predicted_digit = utec::utils::get_predicted_class(logits);

                std::cout << "-------------------------------------\n";
                std::cout << ">>> The AI predicts this digit is: " << predicted_digit << std::endl;
                std::cout << "-------------------------------------\n\n";

            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "A critical error occurred while loading the model: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
