#include <iostream>
#include <vector>
#include <cmath>
// #include "matplotlibcpp.h"
// namespace plt = matplotlibcpp;

using namespace std;

// Definir la función de activación sigmoide
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Definir la función de activación tangente hiperbólica
double tanh_activation(double x) {
    return tanh(x);
}

// Definir la función de activación RELU
double relu(double x) {
    return max(0.0, x);
}

// Definir la función de pérdida Softmax Cross-Entropy
double softmax_cross_entropy_loss(vector<double> x, vector<double> y) {
    double sum_exp = 0.0;
    for (int i = 0; i < x.size(); i++) {
        sum_exp += exp(x[i]);
    }

    double loss = 0.0;
    for (int i = 0; i < x.size(); i++) {
        x[i] = exp(x[i]) / sum_exp; // Aplicar Softmax
        loss += -y[i] * log(x[i] + 1e-10); // Agregar 1e-10 para evitar log(0)
    }

    return loss;
}

// Definir la clase PerceptronMultilayer
class PerceptronMultilayer {
private:
    int num_features;
    int num_hidden_layers;
    int num_neurons_per_layer;
    int num_output_neurons;
    string activation_function;
    vector<vector<vector<double>>> weights;
    vector<vector<double>> biases;
    

public:
vector<double> loss_history;
    // Constructor
    PerceptronMultilayer(int num_features, int num_hidden_layers, int num_neurons_per_layer, int num_output_neurons, string activation_function) {
        // Inicializar los parámetros de la red neuronal
        this->num_features = num_features;
        this->num_hidden_layers = num_hidden_layers;
        this->num_neurons_per_layer = num_neurons_per_layer;
        this->num_output_neurons = num_output_neurons;
        this->activation_function = activation_function;

        // Inicializar los pesos sinápticos
        for (int i = 0; i < num_hidden_layers + 1; i++) {
    vector<vector<double>> layer_weights;
    if (i == 0) {
        layer_weights.resize(num_features);
        for (int j = 0; j < num_features; j++) {
            layer_weights[j].resize(num_neurons_per_layer);
            for (int k = 0; k < num_neurons_per_layer; k++) {
                // Inicialización de Xavier para la capa de entrada
                layer_weights[j][k] = ((double)rand() / RAND_MAX) * sqrt(1.0 / num_features);
            }
        }
    } else if (i == num_hidden_layers) {
        layer_weights.resize(num_neurons_per_layer);
        for (int j = 0; j < num_neurons_per_layer; j++) {
            layer_weights[j].resize(num_output_neurons);
            for (int k = 0; k < num_output_neurons; k++) {
                // Inicialización de Xavier para la capa de salida
                layer_weights[j][k] = ((double)rand() / RAND_MAX) * sqrt(1.0 / num_neurons_per_layer);
            }
        }
    } else {
        layer_weights.resize(num_neurons_per_layer);
        for (int j = 0; j < num_neurons_per_layer; j++) {
            layer_weights[j].resize(num_neurons_per_layer);
            for (int k = 0; k < num_neurons_per_layer; k++) {
                // Inicialización de Xavier para capas ocultas
                layer_weights[j][k] = ((double)rand() / RAND_MAX) * sqrt(1.0 / num_neurons_per_layer);
            }
        }
    }
    weights.push_back(layer_weights);
        }
        // Inicializar los sesgos
        for (int i = 0; i < num_hidden_layers + 1; i++) {
            vector<double> layer_biases;
            if (i == num_hidden_layers) {
                layer_biases.resize(num_output_neurons);
            } else {
                layer_biases.resize(num_neurons_per_layer);
            }
            biases.push_back(layer_biases);
        }
    }

    // Función de entrenamiento
    void train(vector<vector<double>> X, vector<vector<double>> Y, double learning_rate, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epoch_loss = 0.0;
            for (int i = 0; i < X.size(); i++) {
                // Propagación hacia adelante
                vector<vector<double>> activations;
                activations.push_back(X[i]);
                for (int j = 0; j < num_hidden_layers + 1; j++) {
                    vector<double> layer_activations;
                    if (j == num_hidden_layers) {
                        layer_activations.resize(num_output_neurons);
                    } else {
                        layer_activations.resize(num_neurons_per_layer);
                    }
                    for (int k = 0; k < layer_activations.size(); k++) {
                        double z = biases[j][k];
                        for (int l = 0; l < activations[j].size(); l++) {
                            z += activations[j][l] * weights[j][l][k];
                        }
                        if (activation_function == "sigmoid") {
                            layer_activations[k] = sigmoid(z);
                        } else if (activation_function == "tanh") {
                            layer_activations[k] = tanh_activation(z);
                        } else if (activation_function == "relu") {
                            layer_activations[k] = relu(z);
                        }
                    }
                    activations.push_back(layer_activations);
                }

                // Cálculo del error
                vector<double> error;
                for (int j = 0; j < Y[i].size(); j++) {
                    error.push_back(Y[i][j] - activations[num_hidden_layers + 1][j]);
                }

                // Propagación hacia atrás
                vector<vector<double>> deltas;
                deltas.push_back(error);
                for (int j = num_hidden_layers; j >= 0; j--) {
                    vector<double> layer_deltas;
                    if (j == num_hidden_layers) {
                        layer_deltas.resize(num_output_neurons);
                    } else {
                        layer_deltas.resize(num_neurons_per_layer);
                    }
                    for (int k = 0; k < layer_deltas.size(); k++) {
                        double delta = 0.0;
                        if (j == num_hidden_layers) {
                            for (int l = 0; l < deltas[0].size(); l++) {
                                delta += deltas[0][l] * weights[j][l][k];
                            }
                        } else {
                            for (int l = 0; l < deltas[0].size(); l++) {
                                delta += deltas[0][l] * weights[j][l][k];
                            }
                        }
                        if (activation_function == "sigmoid") {
                            layer_deltas[k] = delta * activations[j + 1][k] * (1.0 - activations[j + 1][k]);
                        } else if (activation_function == "tanh") {
                            layer_deltas[k] = delta * (1.0 - pow(activations[j + 1][k], 2));
                        } else if (activation_function == "relu") {
                            if (activations[j + 1][k] > 0.0) {
                                layer_deltas[k] = delta;
                            } else {
                                layer_deltas[k] = 0.0;
                            }
                        }
                    }
                    deltas.insert(deltas.begin(), layer_deltas);
                }

                // Actualización de los pesos sinápticos y los sesgos
                for (int j = 0; j < num_hidden_layers + 1; j++) {
                    for (int k = 0; k < biases[j].size(); k++) {
                        biases[j][k] += learning_rate * deltas[j][k];
                    }
                    for (int k = 0; k < weights[j].size(); k++) {
                        for (int l = 0; l < weights[j][k].size(); l++) {
                            weights[j][k][l] += learning_rate * activations[j][k] * deltas[j + 1][l];
                        }
                    }
                }

                // Calcular la pérdida y agregarla al historial
                epoch_loss += softmax_cross_entropy_loss(activations[num_hidden_layers + 1], Y[i]);
            }

            // Calcular la pérdida promedio para esta época y agregarla al historial
            epoch_loss /= X.size();
            loss_history.push_back(epoch_loss);
        }
    }

    vector<vector<double>> predict(vector<vector<double>> X) {
    vector<vector<double>> predictions;
    for (int i = 0; i < X.size(); i++) {
        // Propagación hacia adelante para cada ejemplo de entrada
        vector<vector<double>> activations;
        activations.push_back(X[i]);
        for (int j = 0; j < num_hidden_layers + 1; j++) {
            vector<double> layer_activations;
            if (j == num_hidden_layers) {
                layer_activations.resize(num_output_neurons);
            } else {
                layer_activations.resize(num_neurons_per_layer);
            }
            for (int k = 0; k < layer_activations.size(); k++) {
                double z = biases[j][k];
                for (int l = 0; l < activations[j].size(); l++) {
                    z += activations[j][l] * weights[j][l][k];
                }
                if (activation_function == "sigmoid") {
                    layer_activations[k] = sigmoid(z);
                } else if (activation_function == "tanh") {
                    layer_activations[k] = tanh_activation(z);
                } else if (activation_function == "relu") {
                    layer_activations[k] = relu(z);
                }
            }
            activations.push_back(layer_activations);
        }
        for (int j = 0; j < activations[num_hidden_layers + 1].size(); j++) {
            activations[num_hidden_layers + 1][j] = (activations[num_hidden_layers + 1][j] > 0.5) ? 1.0 : 0.0;
        }

        predictions.push_back(activations[num_hidden_layers + 1]);
    }
    return predictions;
}

};

