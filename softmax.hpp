#ifndef _SOFTMAX_HPP
#define _SOFTMAX_HPP

#include <vector>
#include <cstdlib>
#include <iostream>
#include <random>
#include <cmath>

using vec1 = std::vector<double>;
using vec2 = std::vector<std::vector<double> >;
using vec3 = std::vector<std::vector<std::vector<double> > >;

class SOFTMAX
{
private:
  int batchSize;
  int input;
  int output;

  vec1 biases;
  vec2 weights;
  vec2 Zs;
  vec2 activations;
  vec2 input_activations;
  vec2 delta;

  void intializeWeights();
  void intializeBiases();
  void softmax(vec2 Zs);

public:
  SOFTMAX(int batchSize, int input, int output);

  void feed(vec2 in);
  void backProp(vec2 out, double eta);


  const vec2 getActivations() const;
  const vec2 getZs() const;
  const vec1 getBiases() const;
  const vec2 getWeights() const;
  const vec2 getDelta() const;

};
#endif
