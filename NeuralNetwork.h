#include <vector>
#include <iostream>
#include <memory>
#include <eigen3/Eigen/Eigen>
#include <chrono>
#include <thread>
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H


using RowVector = Eigen::RowVectorXf;
using Matrix = Eigen::MatrixXf;
using ARowVector = std::shared_ptr<Eigen::RowVectorXf>;
using AMatrix = std::shared_ptr< Eigen::MatrixXf>;


#pragma once
#include <Eigen/Dense>
#include <vector>

class AdamOptimizer {
public:
    AdamOptimizer(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : m_learningRate(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), m_t(0) {}

    void initialize(const std::vector<std::shared_ptr<Matrix>>& weights);

    void update(std::vector<std::shared_ptr<Matrix>>& weights, const std::vector<Eigen::MatrixXf>& gradients);

private:
    float m_learningRate;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    int m_t;

    std::vector<std::shared_ptr<Matrix>> m_m;
    std::vector<std::shared_ptr<Matrix>> m_v;
};

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<long>& layersTopology,const std::string & nameActivationFunction);

    ARowVector forward(const RowVector & input);

    void backward(const RowVector & output,float learningRate);

    void layersError(const RowVector & output);

    void updateWeights(float learningRate);

    void train(const std::vector<ARowVector> & input, const std::vector<ARowVector> & output,float learningRate);

    float classEvaluate(const std::vector<ARowVector>& inputs, const std::vector<ARowVector>& targets);

    float binaryEvaluate(const std::vector<ARowVector>& inputs, const std::vector<ARowVector>& targets);

    void updateWeightsAdam();
    

private:

    std::vector<int64_t> m_layersTopology;
    std::vector<AMatrix> m_weights;

    std::vector<ARowVector> m_layers;
    std::vector<ARowVector> m_errors;
    std::vector<ARowVector> m_cache;

    std::function<float(float)> m_activationFunction;
    std::function<float(float)> m_activationFunctionDerivative;

    AdamOptimizer m_optimizer;
};



#endif //NEURALNETWORK_H
