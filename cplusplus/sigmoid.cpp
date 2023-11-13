#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "funciones.h"
#include "ML.h"
using namespace std;

int main() {
    std::string capas_ocultas, neuronas_por_capa, num_features;
    std::getline(std::cin, capas_ocultas);
    std::getline(std::cin, neuronas_por_capa);
    std::getline(std::cin, num_features);

    int capas_ocultas_int = std::stoi(capas_ocultas);
    int neuronas_por_capa_int = std::stoi(neuronas_por_capa);
    int num_features_int = std::stoi(num_features);

    // leer el train y test
    const char* x_traininfo = "../data_train/x_train.txt";
    vector<vector<double>> x_train;
    leer_x(x_traininfo,x_train);
    const char* y_traininfo = "../data_train/y_train.txt";
    vector<vector<double>> y_train;
    leer_x(y_traininfo,y_train);

    const char* x_testinfo = "../data_test/x_test.txt";
    vector<vector<double>> x_test;
    leer_x(x_testinfo,x_test);

    // caracteristica, numero de capas ocultas, numero de neuronas por capa
    PerceptronMultilayer nn(num_features_int,capas_ocultas_int,neuronas_por_capa_int,1,"sigmoid");
    nn.train(x_train,y_train,0.0001,1000);
    vector<vector<double>> predictions= nn.predict(x_test);
    std::stringstream ss;
    ss << "../output_sigmoid/predictions" << capas_ocultas << "_" << neuronas_por_capa << ".txt";
    std::string archivoSalida1 = ss.str();
    const char* filePath = archivoSalida1.c_str();

    guardar(filePath,predictions);
    
    return 0;
}
