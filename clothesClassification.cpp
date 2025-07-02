#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <complex>
#include <fstream>
#include <memory>
#include "NeuralNetwork.h"
uint64_t CLASSES = 11;
void ReadCSV(const std::string & filename, std::vector<ARowVector>& Xdata,std::vector<ARowVector>& ydata)
{
    std::ifstream file(filename);

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> rowValues;


        while (std::getline(ss, cell, ',')) {
            rowValues.push_back(std::stof(cell));
        }
        RowVector row = Eigen::Map<Eigen::RowVectorXf>(rowValues.data(), rowValues.size());
        auto X = row.block(0,0,1,rowValues.size()-1);
        auto y = row[rowValues.size()-1];
        Xdata.push_back(std::make_shared<RowVector>(X));
        RowVector label(CLASSES);
        label.setZero();
        label[y] = 1.0f;
        ydata.push_back(std::make_shared<RowVector>(label));
    }
}
int main() {
    NeuralNetwork net({1024,64,11},"sigmoid");
    std::vector<ARowVector> Xdata;
    std::vector<ARowVector> ydata;
    ReadCSV("/home/dan/CLionProjects/Desktop/RLTemplate/build/data/clothes_train.csv",Xdata,ydata);
    std::ios::sync_with_stdio(false);


    size_t cutIndex = static_cast<size_t>(std::ceil(Xdata.size() * 0.01));

    std::vector<ARowVector> Xtrain(Xdata.begin(), Xdata.begin() + cutIndex);
    std::vector<ARowVector> ytrain(ydata.begin(), ydata.begin() + cutIndex);

    std::vector<ARowVector> Xtest(Xdata.begin() + cutIndex, Xdata.end());
    std::vector<ARowVector> ytest(ydata.begin() + cutIndex, ydata.end());

    int epochs = 5;
    float learningRate = 0.05f;
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << epochs << std::endl;
        net.train(Xtrain, ytrain, learningRate);
    }
    // net.classEvaluate(Xtest, ytest);
    net.classEvaluate(Xtrain, ytrain);
    return 0;

}
