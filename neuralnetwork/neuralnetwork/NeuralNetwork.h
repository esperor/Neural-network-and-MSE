#pragma once

#include <string>
#include <vector>
#include <iterator>
#include <fstream>
#include <cstdio>

namespace NeuralNetwork {

constexpr double c_e = 2.7182818;
constexpr int c_numberLength = 4;

class Layer;
class Neuron;
class WeightsHandler;

enum class ErrorCode
{
    NoError,
    InvalidNetwork,
    NetworkNotReady,
    InvalidCoefficient,
    InvalidNumberOfSteps,
};

class NeuralNetworkError
{
public:
    NeuralNetworkError(const ErrorCode& errorCode, std::string what = "") : m_errorCode{ errorCode }, m_what{ what } {}

    const std::string& what() const { return m_what; }
    bool operator==(const ErrorCode& code) const { return code == m_errorCode; }

private:
    ErrorCode m_errorCode;
    std::string m_what;
};

class AIntelligence
{
public:
    AIntelligence(WeightsHandler *wHandler) : m_wHandler(wHandler) {}
    AIntelligence(WeightsHandler *wHandler, unsigned short numberOfLayers);
    ~AIntelligence() {}

    void setNumberOfLayers(unsigned short num);
    void setNumberOfNeuronInLayer(unsigned short layer, unsigned short num);
    void setNumberOfInputValues(unsigned short num);
    void setWeightBetweenNeurons(unsigned int i, unsigned int j, float value);

    void writeWeights();
    void updateWHandler();

    bool initializeNetwork();
    bool ready() { return m_ready; }

    std::vector<double> calculateOutput(const std::vector<double> &inputValues);

    unsigned int getNeuronNumber()      { return m_neuronNumber; }
    unsigned int getWeightNumber()      { return m_weightNumber; }
    unsigned short getNumberOfInputs()  { return m_numberOfInputValues; }
    unsigned short getLayersNumber()    { return m_layersNumber;  }

    const std::vector< std::vector<float> > *getWeightLayers();
    const std::vector<float> *getNeuronWeights();
    std::vector<unsigned int> getNeuronNumbersByLayer(short layer);

    float getWeightBetweenNeurons(unsigned int i, unsigned int j);
    std::pair<int, int> getNumberOfNeuronsOfWeightByNumber(unsigned int num);

    // get number of neurons by [layer]
    const std::vector<short> &getNeuronStructure()              { return m_neuronNumberInLayer; }
    // get number of weights by [layer]
    const std::vector<short> &getWeightStructure()              { return m_weightNumberInLayer; }
    const std::vector< std::shared_ptr<Layer> > &getLayers()    { return m_neuronLayers;  }

private:
    bool m_ready = false;

    unsigned short m_layersNumber = 0; // m_neuronLayers.size() == m_layersNumber
    unsigned short m_neuronWeightsNumber = 0;
    unsigned int m_neuronNumber = 0;
    unsigned int m_weightNumber = 0;
    unsigned short m_numberOfInputValues = 0;

    WeightsHandler *m_wHandler = nullptr;

    std::vector<short> m_neuronNumberInLayer;
    std::vector<short> m_weightNumberInLayer;

    std::vector< std::shared_ptr<Layer> > m_neuronLayers;
};

class WeightsHandler
{
public:
    WeightsHandler() {}

    const std::vector< std::vector<float> > *getWeightLayers()  { return &m_weightLayers; }
    const std::vector<float> *getNeuronWeights()                { return &m_neuronWeights; }

    bool readWeights(const std::string &weightsFilename);
    bool ready() { return m_ready; }

    void setWeight(short layer, short num, float value);
    void writeWeights();
    void update();

private:
    bool m_ready = false;
    std::string m_weightsFilename;

    std::vector< std::vector<float> > m_weightLayers;
    std::vector<float> m_neuronWeights;

};

class Layer
{
public:
    Layer(short neuronsNumber);
    ~Layer() {}

    short neuronsNumber() const { return m_neurons.size(); }
    Neuron *getNeuron(short neuronNumber) { return m_neurons.at(neuronNumber).get(); }

private:
    std::vector< std::shared_ptr<Neuron> > m_neurons;

};

class Neuron
{
public:
    Neuron() {}
    ~Neuron() {}

    void calculate(const std::vector<float>& weights, const std::vector<double>& values, float neuronWeight);
    double m_value = 0;
};

}