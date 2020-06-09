#ifndef _SOFTMAX_HPP
#define _SOFTMAX_HPP

#include <vector>
#include <cstdlib>
#include <iostream>
#include <random>
#include <cmath>

class SOFTMAX
{
private:
  int batchSize;
  int input;
  int output;

  std::vector<std::vector<double> > biases;
  std::vector<std::vector<std::vector<double> > > weights;
  std::vector<std::vector<double> > Zs;
  std::vector<std::vector<double> > activations;

  void intializeWeights();
  void intializeBiases();
  void softmax(std::vector<std::vector<double> > Zs);

public:
  SOFTMAX(int batchSize, int input, int output);

  void feed(std::vector<std::vector<double> > input);
  std::vector<std::vector<double> > getActivations() const;

};
#endif
