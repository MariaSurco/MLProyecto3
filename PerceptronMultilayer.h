#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <random>

using namespace std;

template <typename T>
class PerceptronMultilayer
{
private:
    vector<vector<T>> weights_input_hidden;
    vector<vector<T>> weights_hidden_output;

public:
    PerceptronMultilayer(int input_size, int hidden_layers, int hidden_neurons, int output_neurons)
    {

        weights_input_hidden.resize(hidden_layers, vector<T>(hidden_neurons, 0));
        weights_hidden_output.resize(output_neurons, vector<T>(hidden_neurons, 0));

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<T> dist(-1.0, 1.0);

        for (int i = 0; i < hidden_layers; ++i)
        {
            for (int j = 0; j < hidden_neurons; ++j)
            {
                weights_input_hidden[i][j] = dist(gen);
            }
        }

        for (int i = 0; i < output_neurons; ++i)
        {
            for (int j = 0; j < hidden_neurons; ++j)
            {
                weights_hidden_output[i][j] = dist(gen);
            }
        }
    }

    void forward(vector<T> &input)
    {
       
    }

    void train(vector<vector<T>> &input_data, vector<vector<T>> &target_data)
    {
        
    }

    void sigmoid(vector<T> &data) const
    {
        for (int i = 0; i < data.size(); ++i)
        {
            data[i] = 1 / (1 + exp(-data[i]));
        }
    }

    T softmax(vector<T> &predictions, vector<T> &labels) const
    {

        if (predictions.size() != labels.size())
        {
            cerr << "Las dimensiones de las predictionsiones y las etiquetas verdaderas no coinciden." << endl;
            return T(0);
        }

        T max_val = *max_element(predictions.begin(), predictions.end());
        T suma_exp = 0;

        for (int i = 0; i < predictions.size(); i++)
        {
            predictions[i] = exp(predictions[i] - max_val);
            suma_exp += predictions[i];
        }

        for (int i = 0; i < predictions.size(); i++)
        {
            predictions[i] /= suma_exp;
        }

        T loss = T(0);
        for (int i = 0; i < predictions.size(); i++)
        {
            loss += -labels[i] * log(predictions[i]);
        }

        return loss;
    }
};

