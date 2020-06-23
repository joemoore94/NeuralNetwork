#include "sigmoid.hpp"

SIGMOID::SIGMOID() {}

SIGMOID::SIGMOID(int batchSize, int imgX, int imgY, int output, int layers)
{
  this -> batchSize = batchSize;
  this -> imgX = imgX;
  this -> imgY = imgY;
  this -> output = output;
  this -> layers = layers;

  intializeWeights();
  intializeBiases();
}

void SIGMOID::intializeBiases()
{
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0,1.0);

  for (int j = 0; j < output; j++)
  {
    biases.push_back(dist(gen));
  }
}

void SIGMOID::intializeWeights()
{
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0,1.0);

  weights.resize(output);
  for(int h = 0; h < output; h++)
  {
    weights.at(h).resize(layers);
    for (int j = 0; j < layers; j++)
    {
      for (int k = 0; k < imgX*imgY; k++)
      {
        weights.at(h).at(j).push_back(dist(gen));
      }
    }
  }
}


void SIGMOID::feed(vec3 in)
{
  input_activations = in;
  double temp;
  Zs.resize(batchSize);
  activations.resize(batchSize);
  for (int i = 0; i < batchSize; i++)
  {
    Zs.at(i).resize(output);
    activations.at(i).resize(output);
    for (int j = 0; j < output; j++)
    {
      temp = 0;
      for(int h = 0; h < layers; h++)
      {
        for (int k = 0; k < imgX*imgY; k++)
        {
          temp += weights.at(j).at(h).at(k)*in.at(i).at(h).at(k);
        }
      }
      temp += biases.at(j);
      Zs.at(i).at(j) = temp;
      activations.at(i).at(j) = sigmoid(temp);
    }
  }
}

void SIGMOID::backProp(vec2 d, vec2 w, double eta)
{
  vec3& a = input_activations;
  // a is the previous activations, d is the preceding deltas and w is the preceding weights
  delta.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    delta.at(i).resize(output);
    for(int j = 0; j < output; j++)
    {
      for(int k = 0; k < (int)d.at(i).size(); k++)
      {
        delta.at(i).at(j) += w.at(k).at(j)*d.at(i).at(k);
      }
      delta.at(i).at(j) *= sigPrime(Zs.at(i).at(j));
      biases.at(j) -= eta*delta.at(i).at(j)/batchSize;
      for(int l = 0; l < layers; l++)
      {
        for(int m = 0; m < imgX*imgY; m++)
        {
          weights.at(j).at(l).at(m) -= eta*delta.at(i).at(j)*a.at(i).at(l).at(m)/batchSize;
        }
      }
    }
  }
}

double SIGMOID::sigmoid(double z)
{
  return 1 / (1 + std::exp(-z));
}

double SIGMOID::sigPrime(double z)
{
  return sigmoid(z)*(1-sigmoid(z));
}

const vec2 SIGMOID::getActivations() const
{
  return activations;
}

const vec2 SIGMOID::getDelta() const
{
  return delta;
}

const vec3 SIGMOID::getWeights() const
{
  return weights;
}
