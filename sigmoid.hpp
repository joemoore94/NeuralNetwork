#ifndef _SIGMOID_HPP
#define _SIGMOID_HPP

#include <vector>
#include <cstdlib>
#include <iostream>
#include <random>
#include <cmath>

using vec1 = std::vector<double>;
using vec2 = std::vector<std::vector<double> >;
using vec3 = std::vector<std::vector<std::vector<double> > >;

class SIGMOID
{
private:
  int batchSize;
  int imgX;
  int imgY;
  int output;
  int layers;

  vec1 biases;
  vec3 weights;
  vec2 Zs;
  vec2 activations;
  vec3 input_activations;
  vec2 delta;

  double sigmoid(double z);
  double sigPrime(double z);
  void intializeBiases();
  void intializeWeights();

public:
  SIGMOID();
  SIGMOID(int batchSize, int imgX, int imgY, int output, int layers);

  void feed(vec3 in);
  void backProp(vec2 d, vec2 w, double eta);

  const vec2 getActivations() const;
  const vec2 getDelta() const;
  const vec3 getWeights() const;

};
#endif
