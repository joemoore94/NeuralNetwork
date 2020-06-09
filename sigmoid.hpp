#ifndef _SIGMOID_HPP
#define _SIGMOID_HPP

#include <vector>
#include <cstdlib>
#include <iostream>
#include <random>
#include <cmath>

class SIGMOID
{
private:
  int batchSize;
  int imgX;
  int imgY;
  int output;
  int layers;

  std::vector<std::vector<double> > biases;
  std::vector<std::vector<std::vector<std::vector<double> > > > weights;
  std::vector<std::vector<double> > Zs;
  std::vector<std::vector<double> > activations;

  double sigmoid(double z);
  void intializeBiases();
  void intializeWeights();

public:
  SIGMOID(int batchSize, int imgX, int imgY, int output, int layers);

  void feed(std::vector<std::vector<std::vector<double> > > input);
  std::vector<std::vector<double> > getActivations() const;

};
#endif
