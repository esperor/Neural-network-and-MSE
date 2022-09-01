#pragma 

#include <string>
#include "NeuralNetwork.h"

namespace NeuralNetwork
{

class BackPropagation
{
public:
	BackPropagation(AIntelligence *network);
	~BackPropagation() {}

	void setLearningCoef(float n) { m_learningCoef = n; }
	void setInertialCoef(float n) { m_inertialCoef = n; }
	void setNumberOfSteps(int n) { m_numberOfSteps = n; }

	NeuralNetworkError startAlgorithm(const std::vector<double> &inputValues, const std::vector<double> &expectedOutputValues);
	NeuralNetworkError check();

private:
	std::vector<unsigned int> m_outputNeuronsNumbers;
	float m_learningCoef = 0;
	int m_numberOfSteps = 0;
	float m_inertialCoef = 0;
	AIntelligence *m_network;
};

}