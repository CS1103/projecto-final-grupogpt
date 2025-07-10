#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "nn/neural_network.h"
#include "mnist_loader.h"
#include "utils/common_helpers.h"

using namespace utec::neural_network;
namespace fs = std::filesystem;

//── barra de progreso ───────────────────────────────────────────────
void progress(size_t b, size_t total) {
    const int bar = 40;
    int filled = static_cast<int>(bar * b / static_cast<double>(total));
    std::cout << '\r' << '[' << std::string(filled, '#')
              << std::string(bar - filled, ' ')
              << "] " << b << '/' << total << std::flush;
}

//── impresión ASCII (█ / espacio) ───────────────────────────────────
template <typename T>
void print_ascii_bw(const Tensor2D<T>& row784) {
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            T v = row784(0, y * 28 + x);
            if (v <= 1) v *= (T)255;
            std::cout << (v > 128 ? "#" : " ");
        }
        std::cout << '\n';
    }
}

template <typename T>
void inspect_sample(const NeuralNetwork<T>& net,
                    const Tensor2D<T>& X,
                    const Tensor2D<T>& Y,
                    size_t idx)
{
    // crea tensor 1×784
    Tensor2D<T> sample(1, 784);
    for (size_t k = 0; k < 784; ++k)
        sample(0, k) = X(idx, k);

    auto logits = net.forward(sample);
    size_t pred = 0;
    for (size_t j = 1; j < 10; ++j)
        if (logits(0, j) > logits(0, pred)) pred = j;

    size_t label = 0;
    for (size_t j = 1; j < 10; ++j)
        if (Y(idx, j) > Y(idx, label)) label = j;

    std::cout << "\n[Sample " << idx << "]  Pred: " << pred
              << "  Label: " << label << '\n';
    print_ascii_bw(sample);
}

//── guardar / cargar arquitectura y pesos ───────────────────────────
template <typename T>
void save_arch(const NeuralNetwork<T>& net) {
    std::ofstream out("model_architecture.txt");
    for (const auto& l : net.get_layers()) {
        if (auto d = dynamic_cast<Dense<T>*>(l.get()))
            out << "Dense " << d->get_weights().shape()[0] << ' '
                << d->get_weights().shape()[1] << '\n';
        else if (dynamic_cast<ReLU<T>*>(l.get()))
            out << "ReLU\n";
        else
            out << "Unknown\n";
    }
}

template <typename T>
void save_layer_weights(const NeuralNetwork<T>& net) {
    auto& L = net.get_layers();
    dynamic_cast<Dense<T>*>(L[0].get())
        ->save_weights("layer0_weights.txt", "layer0_biases.txt");
    dynamic_cast<Dense<T>*>(L[2].get())
        ->save_weights("layer2_weights.txt", "layer2_biases.txt");
    dynamic_cast<Dense<T>*>(L[4].get())
        ->save_weights("layer4_weights.txt", "layer4_biases.txt");
}

template <typename T>
void load_layer_weights(NeuralNetwork<T>& net) {
    if (!fs::exists("layer0_weights.txt")) return;
    auto& L = net.get_layers();
    dynamic_cast<Dense<T>*>(L[0].get())
        ->load_weights("layer0_weights.txt", "layer0_biases.txt");
    dynamic_cast<Dense<T>*>(L[2].get())
        ->load_weights("layer2_weights.txt", "layer2_biases.txt");
    dynamic_cast<Dense<T>*>(L[4].get())
        ->load_weights("layer4_weights.txt", "layer4_biases.txt");
}

//── main ────────────────────────────────────────────────────────────
int main() {
    try {
        const size_t EPOCHS = 2, BATCH = 100;
        const double LR = 0.001;

        std::cout << "CONFIG  epochs=" << EPOCHS
                  << "  batch="  << BATCH
                  << "  lr="     << LR    << '\n';

        // cargar datos
        auto [X_train, Y_train] = utec::data::load_mnist_csv("mnist_train.csv", 60000);
        auto [X_test , Y_test ] = utec::data::load_mnist_csv("mnist_test.csv" , 10000);

        // construir red
        NeuralNetwork<double> net;
        net.add_layer(std::make_unique<Dense<double>>(784, 128));
        net.add_layer(std::make_unique<ReLU<double>>());
        net.add_layer(std::make_unique<Dense<double>>(128, 64));
        net.add_layer(std::make_unique<ReLU<double>>());
        net.add_layer(std::make_unique<Dense<double>>(64, 10));

        if (fs::exists("layer0_weights.txt")) {
            std::cout << "Loading existing weights...\n";
            load_layer_weights(net);
        }

        SoftmaxCrossEntropyLoss<double> loss;
        Adam<double> opt(LR);
        const size_t N_BATCH = (60000 + BATCH - 1) / BATCH;

        // bucle de epochs
        for (size_t ep = 1; ep <= EPOCHS; ++ep) {
            auto t0 = std::chrono::high_resolution_clock::now();
            double ep_loss = 0.0;

            for (size_t b = 0; b < N_BATCH; ++b) {
                progress(b + 1, N_BATCH);

                size_t n = std::min(BATCH, 60000 - b * BATCH);
                Tensor2D<double> X(n, 784), Y(n, 10);

                // copiar mini‑lote
                for (size_t i = 0; i < n; ++i) {
                    for (size_t k = 0; k < 784; ++k)
                        X(i, k) = X_train(b * BATCH + i, k);
                    for (size_t k = 0; k < 10; ++k)
                        Y(i, k) = Y_train(b * BATCH + i, k);
                }

                ep_loss += loss.forward(net.forward(X), Y);
                net.backward(loss.backward());
                for (auto& l : net.get_layers()) l->update_params(opt);
            }

            double secs = std::chrono::duration<double>(
                            std::chrono::high_resolution_clock::now() - t0).count();

            std::cout << "\nEpoch " << ep << '/' << EPOCHS
                      << "  Loss: " << std::fixed << std::setprecision(4)
                      << ep_loss / N_BATCH << '\n';

            std::cout << "  [Train set] ";
            utec::utils::evaluate(net, X_train, Y_train);

            std::cout << "  [Test  set] ";
            utec::utils::evaluate(net, X_test , Y_test);

            std::cout << "Time: " << secs << " s\n";

            // mostrar 2 aciertos y 1 error
            size_t shown_good = 0;
            bool   shown_bad  = false;
            for (size_t i = 0; i < 10000 && (shown_good < 2 || !shown_bad); ++i) {
                Tensor2D<double> one(std::array<size_t, 2>{1, 784});
                for (size_t k = 0; k < 784; ++k)
                    one(0, k) = X_test(i, k);
                auto logits = net.forward(one);
                size_t pred = 0;
                for (size_t j = 1; j < 10; ++j)
                    if (logits(0, j) > logits(0, pred)) pred = j;
                size_t label = 0;
                for (size_t j = 1; j < 10; ++j)
                    if (Y_test(i, j) > Y_test(i, label)) label = j;
                if (pred == label && shown_good < 2) {
                    std::cout << "\n[Correct " << shown_good + 1
                              << "] Pred: " << pred << "  Label: " << label << '\n';
                    inspect_sample(net, X_test, Y_test, i);
                    ++shown_good;
                }
                else if (pred != label && !shown_bad) {
                    std::cout << "\n[Error]  Pred: " << pred
                              << "  Label: " << label << '\n';
                    inspect_sample(net, X_test, Y_test, i);
                    shown_bad = true;
                }
            }
            std::cout << '\n';
        }

        std::cout << "Saving architecture & weights...\n";
        save_arch(net);
        save_layer_weights(net);
        std::cout << "Done.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Critical error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
