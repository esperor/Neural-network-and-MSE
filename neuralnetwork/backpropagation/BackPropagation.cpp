// BackPropagation.cpp : Defines the functions for the static library.
//

#include "pch.h"
#include "framework.h"
#include "BackPropagation.h"

namespace NeuralNetwork
{

BackPropagation::BackPropagation(AIntelligence *network)
{
    m_network = network;
    m_learningCoef = 0.01;
    m_inertialCoef = 0.01;
}

NeuralNetworkError BackPropagation::check()
{
    if (!m_network)
        return NeuralNetworkError(ErrorCode::InvalidNetwork, "Network is not initialized");
    if (!m_network->ready())
        return NeuralNetworkError(ErrorCode::NetworkNotReady, "Network is not ready");
    if (m_learningCoef >= 1 || m_learningCoef <= 0)
        return NeuralNetworkError(ErrorCode::InvalidCoefficient, "Learning coefficient must be between 0 and 1");
    if (m_inertialCoef >= 1 || m_inertialCoef <= 0)
        return NeuralNetworkError(ErrorCode::InvalidCoefficient, "Inertial coefficient must be between 0 and 1");
    if (m_numberOfSteps <= 0)
        return NeuralNetworkError(ErrorCode::InvalidNumberOfSteps, "Number of steps must be positive");
    return NeuralNetworkError(ErrorCode::NoError);
}

NeuralNetworkError BackPropagation::startAlgorithm
    (const std::vector<double> &inputValues, const std::vector<double> &expectedOutputValues)
{   
    auto sigma = [&](const std::vector<double> &delta, std::vector<unsigned int> neuronNumbers, int j){
        double sum = 0;
        for (auto &k : neuronNumbers)
        {
            sum += (delta.at(k) * m_network->getWeightBetweenNeurons(j, k));
        }
        return sum;
    };

    const short n = 1;

    std::vector< std::vector<double> > weightDelta;
    weightDelta.resize(2);
    weightDelta[0].resize(m_network->getWeightNumber());
    weightDelta[1].resize(m_network->getWeightNumber());
    

    for (int s = 0; s < m_numberOfSteps; s++)
    {
        m_network->calculateOutput(inputValues);
        const auto &layers = m_network->getLayers();
        std::vector<double> delta;
        std::vector<double> o;
        delta.resize(m_network->getNeuronNumber());

        for (int l = 0; l < layers.size(); l++)
        {
            const auto layer = layers[l].get();
            for (int neuron = 0; neuron < layer->neuronsNumber(); neuron++)
            {
                o.push_back(layer->getNeuron(neuron)->m_value);
            }
        }

        m_outputNeuronsNumbers = m_network->getNeuronNumbersByLayer(m_network->getLayersNumber() - 1);
        short neuronsInLayersBefore = m_network->getNeuronNumber() - m_outputNeuronsNumbers.size();

        for (const auto &k : m_outputNeuronsNumbers)
        {
            delta[k] = -o[k] * (1 - o[k]) * (expectedOutputValues[k - neuronsInLayersBefore] - o[k]);
        }

        for (int l = layers.size() - 2; l >= 0; l--)
        {
            neuronsInLayersBefore = 0;
            for (int layer = l - 1; layer >= 0; layer--)
            {
                neuronsInLayersBefore += m_network->getNeuronStructure().at(layer);
            }
            
            for (int j = 0 ; j < m_network->getNeuronStructure().at(l); j++)
            {
                delta[neuronsInLayersBefore + j] = 
                    o[neuronsInLayersBefore + j] * (1 - o[neuronsInLayersBefore + j]) * 
                    sigma(delta, m_network->getNeuronNumbersByLayer(l + 1), neuronsInLayersBefore + j);
            }
        }

        for (int d = 0; d < weightDelta.size(); d++)
        {
            auto pair = m_network->getNumberOfNeuronsOfWeightByNumber(d);
            int i = pair.first;
            int j = pair.second;
            weightDelta[n][d] = m_inertialCoef * weightDelta[n - 1][d] + 
                                (1 - m_inertialCoef) * m_learningCoef * o[i] * delta[j];
            double weight = m_network->getWeightBetweenNeurons(i, j) - weightDelta[n][d];
            m_network->setWeightBetweenNeurons(i, j, weight);
        }

        m_network->writeWeights();
        m_network->updateWHandler();
        weightDelta[n - 1] = weightDelta[n];
    }

    return NeuralNetworkError(ErrorCode::NoError);
}

}
