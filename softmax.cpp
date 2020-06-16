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
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0,1.0);

  weights.resize(output);
  for(int i = 0; i < output; i++)
  {
    weights.at(i).resize(input);
    for(int j = 0; j < input; j++)
    {
      weights.at(i).at(j) = dist(gen);
    }
  }
}

void SOFTMAX::intializeBiases()
{
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0,1.0);

  biases.resize(output);
  for(int i = 0; i < output; i++)
  {
    biases.at(i) = dist(gen);
  }
}

void SOFTMAX::feed(vec2 in)
{
  double temp;
  input_activations = in;
  Zs.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    Zs.at(i).resize(output);
    for(int j = 0; j < output; j++)
    {
      temp = 0;
      for(int k = 0; k < input; k++)
      {
        temp += weights.at(j).at(k)*in.at(i).at(k);
      }
      temp += biases.at(j);
      Zs.at(i).at(j) = temp;
    }
  }
  softmax(Zs);
}

void SOFTMAX::backProp(vec2 out, double eta)
{
  vec2& in = input_activations;
  delta.resize(batchSize); // delta = dC/dZ for each neuron
  for(int i = 0; i < batchSize; i++)
  {
    delta.at(i).resize(output);
    for(int j = 0; j < output; j++)
    {
      delta.at(i).at(j) = (activations.at(i).at(j) - out.at(i).at(j));
      biases.at(j) -= eta*delta.at(i).at(j)/batchSize;
      for(int k = 0; k < input; k++)
      {
        weights.at(j).at(k) -= eta*in.at(i).at(k)*delta.at(i).at(j)/batchSize;
      }
    }
  }
}

void SOFTMAX::softmax(vec2 Zs)
{
  double temp;
  activations.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    activations.at(i).resize(output);
    for(int j = 0; j < output; j++)
    {
      temp = 0;
      for(int k = 0; k < output; k++)
      {
        temp += std::exp(Zs.at(i).at(k));
      }
      temp = std::exp(Zs.at(i).at(j)) / temp;
      activations.at(i).at(j) = temp;
    }
  }
}

const vec2 SOFTMAX::getActivations() const
{
  return activations;
}

const vec2 SOFTMAX::getZs() const
{
  return Zs;
}

const vec1 SOFTMAX::getBiases() const
{
  return biases;
}

const vec2 SOFTMAX::getWeights() const
{
  return weights;
}

const vec2 SOFTMAX::getDelta() const
{
  return delta;
}
