// neuralnetwork.cpp : Defines the functions for the static library.
//

#include "pch.h"
#include "framework.h"
#include "NeuralNetwork.h"

namespace NeuralNetwork {

    AIntelligence::AIntelligence(WeightsHandler *wHandler, unsigned short numberOfLayers)
        : m_wHandler(wHandler)
        , m_layersNumber{ numberOfLayers } {}

    std::vector<double> AIntelligence::calculateOutput(const std::vector<double>& inputValues)
    {
        if (!m_ready)
            throw std::exception("Won't work until ready");

        std::vector<double> layerInputValues = inputValues;
        std::vector<float> layerInputWeights;
        std::vector<double> layerOutputValues;
        for (int layer = 0; layer < m_layersNumber; layer++)
        {
            if (!layerOutputValues.empty())
            {
                layerInputValues = layerOutputValues;
                layerOutputValues.clear();
            }
            for (int i = 0; i < m_neuronNumberInLayer[layer]; i++)
            {
                Neuron* currentNeuron = m_neuronLayers[layer].get()->getNeuron(i);
                layerInputWeights = m_wHandler->getWeightLayers()->at(layer);

                layerInputWeights.erase(layerInputWeights.begin(), std::next(layerInputWeights.begin(), i * layerInputValues.size()));
                layerInputWeights.erase(std::next(layerInputWeights.begin(), layerInputValues.size()), layerInputWeights.end());
                currentNeuron->calculate(layerInputWeights, layerInputValues, m_wHandler->getNeuronWeights()->at(i));
                layerOutputValues.push_back(currentNeuron->m_value);
            }
        }
        return layerOutputValues;
    }

    void AIntelligence::setNumberOfLayers(unsigned short num)
    {
        if (!num)
            throw std::exception("Can't be 0 layers.");

        m_layersNumber = num;
        m_neuronNumberInLayer.resize(num, 0);
        m_weightNumberInLayer.resize(num, 0);
    }

    void AIntelligence::setNumberOfNeuronInLayer(unsigned short layer, unsigned short num)
    {
        if (!num)
            throw std::exception("Can't be 0 neurons in a layer.");

        if (!m_layersNumber)
            setNumberOfLayers(layer);

        m_neuronNumberInLayer[layer] = num;
    }

    void AIntelligence::setNumberOfInputValues(unsigned short num)
    {
        if (!num)
            throw std::exception("Can't be 0 input values.");

        m_numberOfInputValues = num;
    }

    bool AIntelligence::initializeNetwork()
    {
        if (!m_layersNumber || !m_numberOfInputValues || !m_wHandler->ready())
            return false;

        for (int l = 0; l < m_layersNumber; l++)
        {
            short& num = m_neuronNumberInLayer[l];
            if (!num)
                return false;
            else
            {   
                m_neuronLayers.push_back(std::make_shared<Layer>(num));
                m_neuronNumber += num;
                short prev_num;
                if (l == 0)
                    prev_num = m_numberOfInputValues;
                else
                    prev_num = m_neuronNumberInLayer[l - 1];

                m_weightNumberInLayer[l] = prev_num * num;
                m_weightNumber += m_weightNumberInLayer[l];
            }
        }
        m_neuronWeightsNumber = m_neuronNumber;

        m_ready = true;
        return true;
    }

    const std::vector< std::vector<float> > *AIntelligence::getWeightLayers()
    {
        try 
        {
            return m_wHandler->getWeightLayers();
        }
        catch (...) 
        { 
            return nullptr;
        }
    }

    const std::vector<float> *AIntelligence::getNeuronWeights()
    {
        try 
        {
            return m_wHandler->getNeuronWeights();
        }
        catch (...) 
        { 
            return nullptr;
        }
    }

    std::vector<unsigned int> AIntelligence::getNeuronNumbersByLayer(short layer)
    {
        unsigned int neuronsBefore = 0;
        for (int i = 0; i < layer; i++)
            neuronsBefore += m_neuronNumberInLayer[i];

        std::vector<unsigned int> neuronNumbers;
        for (int i = 0; i < m_neuronNumberInLayer[layer]; i++)
        {
            neuronNumbers.push_back(neuronsBefore + i);
        }
        return neuronNumbers;
    }

    float AIntelligence::getWeightBetweenNeurons(unsigned int i, unsigned int j)
    {

        auto findLayer = [&](short& neuronNumberInItsLayer, unsigned int num) {
            unsigned int neuronNumberInLayersBefore = 0;
            for (int l = 0; l < m_neuronNumberInLayer.size(); l++)
            {
                if (num < neuronNumberInLayersBefore + m_neuronNumberInLayer[l])
                {
                    neuronNumberInItsLayer = num - neuronNumberInLayersBefore;
                    return l;
                }
                neuronNumberInLayersBefore += m_neuronNumberInLayer[l];
            }
            return -1;
        };

        short layerOfFirstNeuron, layerOfSecondNeuron;
        short firstNeuronNumberInItsLayer, secondNeuronNumberInItsLayer;

        layerOfFirstNeuron = findLayer(firstNeuronNumberInItsLayer, i);
        if (layerOfFirstNeuron == -1)
            throw std::exception(("Invalid first neuron number: " + std::to_string(i) +
                "is bigger than number of all neurons (" + std::to_string(m_neuronNumber) + ")").c_str());

        layerOfSecondNeuron = findLayer(secondNeuronNumberInItsLayer, j);
        if (layerOfSecondNeuron == -1)
            throw std::exception(("Invalid first neuron number: " + std::to_string(j) +
                "is bigger than number of all neurons (" + std::to_string(m_neuronNumber) + ")").c_str());

        return m_wHandler->getWeightLayers()->at(layerOfFirstNeuron + 1)
            .at(firstNeuronNumberInItsLayer * m_neuronNumberInLayer[layerOfFirstNeuron + 1] + secondNeuronNumberInItsLayer);
    }

    bool WeightsHandler::readWeights(const std::string& weightsFilename)
    {
        auto readValue = [](const std::string& line, int i) {
            // (6 is the number of symbols for one value and 4 is the number of significant symbols for one value)
            return std::stof(line.substr(i, c_numberLength).c_str());
        };

        m_weightsFilename = weightsFilename;

        std::ifstream weightstream(m_weightsFilename);
        if (weightstream) {
            std::string line;
            std::vector<std::string> lines;
            while (std::getline(weightstream, line))
            {
                if (line[0] == '#')
                    continue;

                lines.push_back(line);
            }

            m_weightLayers.resize(lines.size());

            for (int l = 0; l < lines.size() - 1; l++)
                for (int c = 0; c < lines[l].size(); c += c_numberLength + 2) // 2 is for comma and space
                    // reading one value at a time 
                    m_weightLayers[l].push_back(readValue(lines[l], c));

            for (int c = 0; c < lines[lines.size() - 1].size(); c += c_numberLength + 2)
                // reading one value at a time
                m_neuronWeights.push_back(readValue(lines[lines.size() - 1], c));
        }
        else
            return false;

        m_ready = true;
        return true;
    }

    std::pair<int, int> AIntelligence::getNumberOfNeuronsOfWeightByNumber(unsigned int num)
    {
        short layer = 0;
        unsigned int weightNumberInLayersBefore = 0;
        while (layer < m_weightNumberInLayer.size())
        {
            if (num < weightNumberInLayersBefore + m_weightNumberInLayer[layer])
                break;
            weightNumberInLayersBefore += m_neuronNumberInLayer[layer];
            layer++;
        }
        int numberOfNeuronsInPrevLayer = m_neuronNumberInLayer[layer];
        int numberOfNeuronsInNextLayer = m_neuronNumberInLayer[layer + 1];
        int second = num % numberOfNeuronsInNextLayer + numberOfNeuronsInPrevLayer;
        int first = num / (numberOfNeuronsInNextLayer - second);
        return std::pair<int, int>(first, second);
    }


    void WeightsHandler::setWeight(short layer, short num, float value)
    {
        m_weightLayers[layer][num] = value;
    }

    void WeightsHandler::writeWeights()
    {
        remove(m_weightsFilename.c_str());
        std::ofstream weightstream(m_weightsFilename, std::ofstream::out | std::ofstream::trunc);
        if (weightstream)
        {
            for (const auto& layer : m_weightLayers)
            {   
                std::string line = "";
                for (const auto& value : layer)
                {
                    line += std::to_string(value).substr(0, c_numberLength) + ", ";
                }
                weightstream << line << std::endl;
            }

            std::string line = "";
            for (const auto& value : m_neuronWeights)
            {
                line += std::to_string(value).substr(0, c_numberLength) + ", ";
            }
            weightstream << line << std::endl;
        }
        else
            throw std::exception("Couldn't write to file");

        weightstream.close();

    }

    void WeightsHandler::update()
    {
        m_weightLayers.clear();
        m_neuronWeights.clear();
        readWeights(m_weightsFilename);
    }

    void AIntelligence::writeWeights()
    {
        m_wHandler->writeWeights();
    }

    void AIntelligence::updateWHandler()
    {
        m_wHandler->update();
    }

    void AIntelligence::setWeightBetweenNeurons(unsigned int i, unsigned int j, float value)
    {
        unsigned int neuronNumberInLayersBefore = 0;

        auto findLayer = [&](short& neuronNumberInItsLayer, unsigned int num) {
            unsigned int neuronNumberInLayersBefore = 0;
            for (int l = 0; l < m_neuronNumberInLayer.size(); l++)
            {
                if (num < neuronNumberInLayersBefore + m_neuronNumberInLayer[l])
                {
                    neuronNumberInItsLayer = num - neuronNumberInLayersBefore;
                    return l;
                }
                neuronNumberInLayersBefore += m_neuronNumberInLayer[l];
            }
            return -1;
        };

        short layerOfFirstNeuron, layerOfSecondNeuron;
        short firstNeuronNumberInItsLayer, secondNeuronNumberInItsLayer;

        layerOfFirstNeuron = findLayer(firstNeuronNumberInItsLayer, i);
        if (layerOfFirstNeuron == -1)
            throw std::exception(("Invalid first neuron number: " + std::to_string(i) +
                "is bigger than number of all neurons (" + std::to_string(m_neuronNumber) + ")").c_str());

        layerOfSecondNeuron = findLayer(secondNeuronNumberInItsLayer, j);
        if (layerOfSecondNeuron == -1)
            throw std::exception(("Invalid first neuron number: " + std::to_string(i) +
                "is bigger than number of all neurons (" + std::to_string(m_neuronNumber) + ")").c_str());

        short layer = layerOfFirstNeuron + 1;
        short num = firstNeuronNumberInItsLayer * m_neuronNumberInLayer[layerOfFirstNeuron + 1] + secondNeuronNumberInItsLayer;
        m_wHandler->setWeight(layer, num, value);
    }


    Layer::Layer(short neuronsNumber)
    {
        for (int i = 0; i < neuronsNumber; i++)
        {
            m_neurons.push_back(std::make_shared<Neuron>());
        }
    }


    void Neuron::calculate(const std::vector<float>& weights, const std::vector<double>& values, float neuronWeight)
    {
        if (weights.size() != values.size())
            throw std::runtime_error("Number of weights must match number of values");

        std::vector<float> zipped;
        for (unsigned int i = 0; i < weights.size(); i++)
        {
            zipped.push_back(weights[i] * values[i]);
        }

        double sum = 0;
        for (auto element : zipped)
            sum += element;

        m_value = 1 / (1 + std::pow(c_e, -(sum)+neuronWeight));
    }

}
