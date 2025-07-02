//
// Created by dan on 6/26/25.
//
// #define EIGEN_DEFAULT_DENSE_INDEX_TYPE = size_t

#include "NeuralNetwork.h"
#include <iomanip>
RowVector softmax(const RowVector& z) {
    RowVector expZ = z.array().exp();
    float sum = expZ.sum();
    return expZ / sum;
}

void AdamOptimizer::initialize(const std::vector<std::shared_ptr<Matrix>> &weights) {
    m_m.clear();
    m_v.clear();
    for (const auto& w : weights) {
        m_m.push_back(std::make_shared<Matrix>(Matrix::Zero(w->rows(), w->cols())));
        m_v.push_back(std::make_shared<Matrix>(Matrix::Zero(w->rows(), w->cols())));
    }
    m_t = 0;
}

void AdamOptimizer::update(std::vector<std::shared_ptr<Matrix>> &weights,
    const std::vector<Eigen::MatrixXf> &gradients) {
    m_t++;

    for (size_t i = 0; i < weights.size(); ++i) {
        auto& m = *m_m[i];
        auto& v = *m_v[i];
        const auto& g = gradients[i];

        m = m_beta1 * m + (1.0f - m_beta1) * g;
        v = m_beta2 * v + (1.0f - m_beta2) * g.cwiseProduct(g);

        Matrix m_hat = m / (1.0f - std::pow(m_beta1, m_t));
        Matrix v_hat = v / (1.0f - std::pow(m_beta2, m_t));

        for (int r = 0; r < weights[i]->rows(); ++r) {
            for (int c = 0; c < weights[i]->cols(); ++c) {
                (*weights[i])(r, c) += m_learningRate * m_hat(r, c) / (std::sqrt(v_hat(r, c)) + m_epsilon);
            }
        }
    }
}

NeuralNetwork::NeuralNetwork(const std::vector<int64_t> &layersTopology, const std::string &nameActivationFunction) {
    m_layersTopology = layersTopology;

    if (nameActivationFunction == "tanh") {
        m_activationFunction = std::tanhf;
        m_activationFunctionDerivative = std::function<double(double)>([&](const double x) {return 1 - tanh(x) * tanh(x);});
    } else if (nameActivationFunction == "relu") {
        m_activationFunction =           std::function<double(double)>([&](const double x) {return std::max(0.0, x); });
        m_activationFunctionDerivative = std::function<double(double)>([&](const double x) {return x >= 0 ? 1.0 : 0.0;});
    } else if (nameActivationFunction == "sigmoid") {
        m_activationFunction =           std::function<double(double)>([&](const double x) { return 1 / (1 + exp(-x)); });
        m_activationFunctionDerivative = std::function<double(double)>([&](const double x) {
            double tmp = 1 / (1 + exp(-x));
            return tmp * (1 - tmp);
        });
    }

    for (size_t i = 0; i < m_layersTopology.size(); i++) {
        if (i == m_layersTopology.size() - 1) {
            m_layers.push_back(std::make_shared<RowVector>(m_layersTopology[i]));
            m_cache.push_back(std::make_shared<RowVector>(m_layersTopology[i]));
        } else {
            m_layers.push_back(std::make_shared<RowVector>(m_layersTopology[i] + 1));
            m_cache.push_back(std::make_shared<RowVector>(m_layersTopology[i] + 1));
        }
        m_errors.push_back(std::make_shared<RowVector>(m_cache.back()->size()));
        if (i != m_layersTopology.size() - 1) {
            m_layers.back()->coeffRef(m_layersTopology[i]) = 1.0;
            m_cache.back()->coeffRef(m_layersTopology[i]) = 1.0;
        }
        if (i > 0) {
            if (i != m_layersTopology.size() - 1) {
                m_weights.push_back(std::make_shared<Matrix>(m_layersTopology[i - 1] + 1, m_layersTopology[i] + 1));
                m_weights.back()->setRandom();
                m_weights.back()->col(m_layersTopology[i]).setZero();
                m_weights.back()->coeffRef(m_layersTopology[i - 1], m_layersTopology[i]) = 1.0;
            } else {
                m_weights.push_back(std::make_shared<Matrix>(m_layersTopology[i - 1] + 1, m_layersTopology[i]));
                m_weights.back()->setRandom();
            }
        }
    }
    m_optimizer.initialize(m_weights);
}

ARowVector NeuralNetwork::forward(const RowVector &input) {
    m_layers.front()->block(0, 0, 1, input.size()) = input;

    for (size_t i = 1; i < m_layers.size(); ++i) {
        *m_cache[i] = (*m_layers[i - 1]) * (*m_weights[i - 1]);

        if (i == m_layers.size() - 1) {

            *m_layers[i] = softmax(*m_cache[i]);
        } else {
            const int64_t limit = m_layersTopology[i];
            for (int64_t j = 0; j < limit; ++j) {
                m_layers[i]->coeffRef(j) = m_activationFunction(m_cache[i]->coeffRef(j));
            }
            m_layers[i]->coeffRef(m_layersTopology[i]) = 1.0;
            m_cache[i]->coeffRef(m_layersTopology[i]) = 1.0;
        }
    }

    return m_layers.back();
}

void NeuralNetwork::backward(const RowVector &output, const float learningRate) {
    layersError(output);
    // updateWeights(learningRate);
    updateWeightsAdam();
}

void NeuralNetwork::layersError(const RowVector &output) {
    *m_errors.back() = output - *m_layers.back();

    for (int i = static_cast<int>(m_layersTopology.size()) - 2; i > 0; --i) {
        *m_errors[i] = (*m_errors[i + 1]) * m_weights[i]->transpose();
        for (int64_t j = 0; j < m_layersTopology[i]; ++j)
            m_errors[i]->coeffRef(j) *= m_activationFunctionDerivative(m_cache[i]->coeffRef(j));

        m_errors[i]->coeffRef(m_layersTopology[i]) = 0.0;
    }
}

void NeuralNetwork::updateWeights(const float learningRate) {
    for (int64_t i = 0; i < m_layersTopology.size() - 1; ++i) {
        const int64_t rows = m_weights[i]->rows();
        const int64_t cols = m_weights[i]->cols();

        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                float error = m_errors[i + 1]->coeffRef(c);
                float deriv = m_activationFunctionDerivative(m_cache[i + 1]->coeffRef(c));
                float input = m_layers[i]->coeffRef(r);

                m_weights[i]->coeffRef(r, c) += learningRate * error * deriv * input;
            }
        }
    }
}

void NeuralNetwork::train(const std::vector<ARowVector> &input,
                          const std::vector<ARowVector> &output,
                          const float learningRate) {


    std::string upperProgressBar = "╔════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::string middleProgressBar= "║....................................................................................................║";
    std::string lowerProgressBar = "╚════════════════════════════════════════════════════════════════════════════════════════════════════╝\n";

    std::string bar = upperProgressBar + middleProgressBar +"\n" + lowerProgressBar;
    std::cout << bar;
    std::cout << "\033[2A" << "\033[1C" <<std::flush;
    size_t progressBarIndex = 0;
    const size_t n = input.size();
    double mod = static_cast<double>(n) / 100;
    for (size_t i = 0; i < input.size(); i++) {
            forward(*input[i]);
            backward(*output[i], learningRate);

        if (i >= mod * progressBarIndex) {
            middleProgressBar[progressBarIndex+3] = '#';
            progressBarIndex++;
            std::cout << "\r" << middleProgressBar << "\t" <<std::setprecision(2)<< std::setw(3)<<std::fixed <<static_cast<double>(i) / static_cast<double>(n) * 100.0f << " %" <<std::flush;
        }
    }
    std::cout << "\r" << middleProgressBar  << "\t"<< std::setw(3) << std::setprecision(2) << 100.0f << " %" <<std::flush;
    std::cout << "\033[2B\r";
}

float NeuralNetwork::classEvaluate(const std::vector<ARowVector> &inputs, const std::vector<ARowVector> &targets) {
    int correct = 0;

    for (int64_t i = 0; i < inputs.size(); ++i) {
        auto pred = this->forward(*inputs[i]);

        Eigen::Index predicted_class;
        pred->maxCoeff(&predicted_class);

        Eigen::Index label;
        targets[i]->maxCoeff(&label);

        if (predicted_class == label) {
            correct++;
        }
    }
    const float acc = 100.0f * correct / inputs.size();
    std::cout << "Test Accuracy: " << acc << "\n";
    return acc;
}

float NeuralNetwork::binaryEvaluate(const std::vector<ARowVector> &inputs, const std::vector<ARowVector> &targets) {
    int correct = 0;
    float totalLoss = 0.0f;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto pred = forward(*inputs[i]);
        float output = (*pred)(0);
        float target = (*targets[i])(0);
        float error = output - target;
        totalLoss += error * error;

        if ((output >= 0.5f && target == 1.0f) || (output < 0.5f && target == 0.0f))
            correct++;
    }
    float acc = 100.0f * correct / inputs.size();
    float mse = totalLoss / inputs.size();
    std::cout << "Test Accuracy: " << acc << "%\tMSE: " << mse << "\n";
    return acc;
}

void NeuralNetwork::updateWeightsAdam() {
    std::vector<Eigen::MatrixXf> gradients;

    for (uint i = 0; i < m_layersTopology.size() - 1; i++) {
        Eigen::MatrixXf grad(m_weights[i]->rows(), m_weights[i]->cols());

        for (uint r = 0; r < m_weights[i]->rows(); r++) {
            for (uint c = 0; c < m_weights[i]->cols(); c++) {
                float d_activation = m_activationFunctionDerivative(m_cache[i + 1]->coeffRef(c));
                grad(r, c) = m_errors[i + 1]->coeffRef(c) * d_activation * m_layers[i]->coeffRef(r);

            }
        }

        gradients.push_back(grad);
    }

    m_optimizer.update(m_weights, gradients);
}
