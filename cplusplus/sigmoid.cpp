#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "funciones.h"
#include "ML.h"
using namespace std;



int main() {
    // leer el trina y test
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
    PerceptronMultilayer nn(10,1,3,1,"sigmoid");
    nn.train(x_train,y_train,0.0001,1000);
    vector<vector<double>> predictions1_3= nn.predict(x_test);
    const char* archivoSalida1 = "../output_sigmoid/predictions1_3.txt";
    guardar(archivoSalida1,predictions1_3);


    PerceptronMultilayer nn2(10,1,2,1,"sigmoid");
    nn.train(x_train,y_train,0.0001,1000);
    vector<vector<double>> predictions1_2= nn2.predict(x_test);
    const char* archivoSalida2 = "../output_sigmoid/predictions1_2.txt";
    guardar(archivoSalida2,predictions1_2);

    PerceptronMultilayer nn3(10,3,6,1,"sigmoid");
    nn.train(x_train,y_train,0.0001,1000);
    vector<vector<double>> predictions3_6= nn3.predict(x_test);
    const char* archivoSalida3 = "../output_sigmoid/predictions3_6.txt";
    guardar(archivoSalida3,predictions3_6);

    PerceptronMultilayer nn4(10,4,6,1,"sigmoid");
    nn.train(x_train,y_train,0.0001,1000);
    vector<vector<double>> predictions4_6= nn4.predict(x_test);
    const char* archivoSalida4 = "../output_sigmoid/predictions4_6.txt";
    guardar(archivoSalida4,predictions4_6);

    PerceptronMultilayer nn5(10,10,8,1,"sigmoid");
    nn.train(x_train,y_train,0.0001,1000);
    vector<vector<double>> predictions10_8= nn5.predict(x_test);
    const char* archivoSalida5 = "../output_sigmoid/predictions10_8.txt";
    guardar(archivoSalida5,predictions10_8);



    
    return 0;
}
