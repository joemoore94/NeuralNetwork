#include "softmax.hpp"

SOFTMAX::SOFTMAX(int batchSize, int input, int output)
{
  this -> batchSize = batchSize;
  this -> input = input;
  this -> output = output;

  intializeWeights();
  intializeBiases();
}

void SOFTMAX::intializeWeights()
{
  weights.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    weights.at(i).resize(output);
    for(int j = 0; j < output; j++)
    {
      for(int k = 0; k < input; k++)
      {
        weights.at(i).at(j).push_back(0);
      }
    }
  }
}

void SOFTMAX::intializeBiases()
{
  biases.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    for(int j = 0; j < output; j++)
    {
      biases.at(i).push_back(0);
    }
  }
}

void SOFTMAX::feed(std::vector<std::vector<double> > input)
{
  double temp;
  Zs.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    for(int j = 0; j < output; j++)
    {
      temp = 0;
      for(int k = 0; k < this -> input; k++)
      {
        temp += weights.at(i).at(j).at(k)*input.at(i).at(k);
      }
      temp += biases.at(i).at(j);
      Zs.at(i).push_back(temp);
    }
  }
  softmax(Zs);
}

void SOFTMAX::softmax(std::vector<std::vector<double> > Zs)
{
  double temp;
  activations.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    for(int j = 0; j < output; j++)
    {
      temp = 0;
      for(int k = 0; k < output; k++)
      {
        temp += std::exp(Zs.at(i).at(k));
      }
      temp = Zs.at(i).at(j) / temp;
      activations.at(i).push_back(temp);
    }
  }
}

std::vector<std::vector<double> > SOFTMAX::getActivations() const
{
  return activations;
}
