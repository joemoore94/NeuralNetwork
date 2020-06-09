#include "sigmoid.hpp"

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

void SIGMOID::feed(std::vector<std::vector<std::vector<double> > > input)
{
  double temp;
  Zs.resize(batchSize);
  activations.resize(batchSize);
  for (int i = 0; i < batchSize; i++)
  {
    for (int j = 0; j < output; j++)
    {
      temp = 0;
      for(int h = 0; h < layers; h++)
      {
        for (int k = 0; k < imgX*imgY; k++)
        {
          temp += weights.at(i).at(j).at(h).at(k)*input.at(i).at(h).at(k);
        }
      }
      temp += biases.at(i).at(j);
      Zs.at(i).push_back(temp);
      activations.at(i).push_back(sigmoid(temp));
    }
  }
}

void SIGMOID::intializeBiases()
{
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0,1.0);

  biases.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    for (int j = 0; j < output; j++)
    {
      biases.at(i).push_back(dist(gen));
    }
  }
}

void SIGMOID::intializeWeights()
{
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0,1.0);

  weights.resize(batchSize);
  for (int i = 0; i < batchSize; i++)
  {
    weights.at(i).resize(output);
    for(int h = 0; h < output; h++)
    {
      weights.at(i).at(h).resize(layers);
      for (int j = 0; j < layers; j++)
      {
        weights.at(i).at(h).at(j).resize(imgX*imgY);
        for (int k = 0; k < imgX*imgY; k++)
        {
          weights.at(i).at(h).at(j).push_back(dist(gen));
        }
      }
    }
  }
}

double SIGMOID::sigmoid(double z)
{
  return 1 / (1 + std::exp(-z));
}

std::vector<std::vector<double> > SIGMOID::getActivations() const
{
  return activations;
}
