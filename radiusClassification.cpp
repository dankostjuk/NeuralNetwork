#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include "NeuralNetwork.h"



const float RADIUS = 0.8f;
const float RADIUS_SQ = RADIUS * RADIUS;

RowVector classify(float x, float y) {
    return (x * x + y * y > RADIUS_SQ) ? RowVector({{0.0f,1.0f}}) :RowVector ({{1.0f,0.0f}});
}

void generateData(std::vector<ARowVector>& inputs, std::vector<ARowVector>& targets, int numSamples, std::mt19937& gen) {
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    for (int i = 0; i < numSamples; ++i) {
        float x = dist(gen);
        float y = dist(gen);
        auto input = std::make_shared<RowVector>(RowVector{{x, y}});
        auto target = std::make_shared<RowVector>(classify(x, y));
        inputs.push_back(input);
        targets.push_back(target);
    }
}


int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<ARowVector> trainInputs, trainTargets;
    std::vector<ARowVector> testInputs, testTargets;

    generateData(trainInputs, trainTargets, 10000, gen);
    generateData(testInputs, testTargets, 1000, gen);

    NeuralNetwork net({2, 5, 5, 2},"sigmoid");
    std::ios::sync_with_stdio(false);

    int epochs = 10;
    float learningRate = 0.05f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << epochs << std::endl;
        net.train(trainInputs, trainTargets, learningRate);
        net.classEvaluate(testInputs, testTargets);
    }

    std::cout << "\n=== Final Evaluation ===\n";
    net.classEvaluate(testInputs, testTargets);

    std::cout << "\n=== Sample Predictions ===\n";
    for (int i = 0; i < 10; ++i) {
        auto input = testInputs[i];
        float x = (*input)(0);
        float y = (*input)(1);
        auto pred = net.forward(*input);

        Eigen::Index predicted_class;
        pred->maxCoeff(&predicted_class);

        Eigen::Index label;
        testTargets[i]->maxCoeff(&label);

        std::cout << std::setprecision(3)  << std::fixed;
        std::cout <<  "In: ["  << std::setw(8) << x << ", \t" << std::setw(8) << y << "] \tTarget: " << label << " Pred: " << predicted_class << " Distance: "  <<std::sqrt(x*x+y*y)<< std::endl;
    }

    return 0;
}
