#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;

int leer_x(const char* archiv, vector<vector<double>>& x){
    ifstream archivoEntrada(archiv);
    if (!archivoEntrada.is_open()) {
        cerr << "Error al abrir el archivo " << archiv << endl;
        return 1;
    }
    // Leer línea por línea
    string linea;
    while (getline(archivoEntrada, linea)) {
        // Crear un stringstream desde la línea
        stringstream ss(linea);
        
        // Vector temporal para almacenar los valores de la línea
        vector<double> valores;

        // Leer valores separados por comas desde el stringstream
        double valor;
        char coma;  // Para almacenar las comas
        while (ss >> valor) {
            valores.push_back(valor);
            if (ss >> coma && coma != ',') {
                cerr << "Error: Se esperaba una coma después del valor." << endl;
                return 1;
            }
        }

        // Agregar el vector de valores al vector principal
        x.push_back(valores);
    }
    // Cerrar el archivo
    archivoEntrada.close();
    return 0;

}

int leer_y(const char* archivo, vector<int>& y) {
    ifstream archivoEntrada(archivo);
    if (!archivoEntrada.is_open()) {
        cerr << "Error al abrir el archivo " << archivo << endl;
        return 1;
    }

    // Leer línea por línea
    string linea;
    while (getline(archivoEntrada, linea)) {
        // Crear un stringstream desde la línea
        stringstream ss(linea);

        // Valor para almacenar el único valor de la línea
        int valor;

        // Leer el único valor desde el stringstream
        if (!(ss >> valor)) {
            cerr << "Error: No se pudo leer el valor desde la línea." << endl;
            return 1;
        }

        // Agregar el valor al vector principal
        y.push_back(valor);
    }

    // Cerrar el archivo
    archivoEntrada.close();
    return 0;
}
void guardar(const char* rutaArchivo, const std::vector<std::vector<double>>& matriz) {
    // Abrir el archivo de salida
    std::ofstream archivo(rutaArchivo);
    if (!archivo.is_open()) {
        std::cerr << "Error al abrir el archivo " << rutaArchivo << std::endl;
        return;
    }

    // Iterar sobre el vector de vectores y escribir en el archivo
    for (const auto& fila : matriz) {
        for (size_t i = 0; i < fila.size(); ++i) {
            archivo << fila[i];
            // Si no es el último elemento de la fila, añadir una coma
            if (i < fila.size() - 1) {
                archivo << ",";
            }
        }
        // Terminar la línea después de cada fila
        archivo << std::endl;
    }

    // Cerrar el archivo
    archivo.close();
}