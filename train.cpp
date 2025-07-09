//
// Created by migue on 6/07/2025.
//

#include <iostream>
#include <vector>
#include <iomanip>
#include "neural_network.h"
#include "mnist_loader.h"
#include "common_helpers.h"

using namespace utec::neural_network;

int main() {
    try {
        const std::string TRAIN_CSV = "mnist_train.csv";
        const std::string TEST_CSV = "mnist_test.csv";
        const size_t NUM_TRAIN = 60000;
        const size_t NUM_TEST = 10000;
        const size_t IMAGE_SIZE = 784;
        const size_t NUM_CLASSES = 10;

        const size_t EPOCHS = 25;
        const size_t BATCH_SIZE = 100;
        const double LEARNING_RATE = 0.0005;

        std::cout << "Loading MNIST data..." << std::endl;
        auto [train_images, train_labels] = utec::data::load_mnist_csv(TRAIN_CSV, NUM_TRAIN);
        auto [test_images, test_labels] = utec::data::load_mnist_csv(TEST_CSV, NUM_TEST);
        std::cout << "Data loaded." << std::endl;

        NeuralNetwork<double> model;
        model.add_layer(std::make_unique<Dense<double>>(IMAGE_SIZE, 256));
        model.add_layer(std::make_unique<ReLU<double>>());
        model.add_layer(std::make_unique<Dense<double>>(256, 128));
        model.add_layer(std::make_unique<ReLU<double>>());
        model.add_layer(std::make_unique<Dense<double>>(128, NUM_CLASSES));

        SoftmaxCrossEntropyLoss<double> criterion;

        std::cout << "\nStarting training..." << std::endl;
        size_t num_batches = NUM_TRAIN / BATCH_SIZE;

        for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
            double epoch_loss = 0.0;
            for (size_t i = 0; i < num_batches; ++i) {
                Tensor2D<double> X_batch(BATCH_SIZE, IMAGE_SIZE);
                Tensor2D<double> Y_batch(BATCH_SIZE, NUM_CLASSES);
                for(size_t j = 0; j < BATCH_SIZE; ++j) {
                    for(size_t k = 0; k < IMAGE_SIZE; ++k) X_batch(j, k) = train_images(i * BATCH_SIZE + j, k);
                    for(size_t k = 0; k < NUM_CLASSES; ++k) Y_batch(j, k) = train_labels(i * BATCH_SIZE + j, k);
                }

                auto logits = model.forward(X_batch);
                epoch_loss += criterion.forward(logits, Y_batch);
                auto grad = criterion.backward();
                model.backward(grad);

                for (auto& layer : model.get_layers()) {
                     layer->update(LEARNING_RATE);
                }
            }
            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                      << ", Loss: " << std::fixed << std::setprecision(4) << epoch_loss / num_batches << ", ";
            utec::utils::evaluate(model, test_images, test_labels);
        }

        std::cout << "\nTraining finished." << std::endl;

        std::cout << "Saving model weights..." << std::endl;
        dynamic_cast<Dense<double>*>(model.get_layers()[0].get())->save_weights("layer0_weights.txt", "layer0_biases.txt");
        dynamic_cast<Dense<double>*>(model.get_layers()[2].get())->save_weights("layer2_weights.txt", "layer2_biases.txt");
        dynamic_cast<Dense<double>*>(model.get_layers()[4].get())->save_weights("layer4_weights.txt", "layer4_biases.txt");
        std::cout << "Weights saved successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "A critical error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
