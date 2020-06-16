#ifndef _CONVOLUTION_HPP
#define _CONVOLUTION_HPP

#include <vector>
#include <cstdlib>
#include <iostream>
#include <random>

using vec1 = std::vector<double>;
using vec2 = std::vector<std::vector<double> >;
using vec3 = std::vector<std::vector<std::vector<double> > >;

class CONVOLUTION
{
private:
  int batchSize;
  int imgX;
  int imgY;
  int conX;
  int conY;
  int layers;
  int num = 0;

  vec1 biases;
  vec2 weights;
  vec3 Zs;
  vec3 activations;
  vec2 input_Activations;
  vec3 input_activations;
  vec3 delta;

  double ReLU(double z);
  double ReLUP(double z);
  void intializeBiases();
  void intializeWeights();

public:
  CONVOLUTION(int batchSize, int imgX, int imgY, int conX, int conY, int layers);

  void feed(vec2 in);
  void feed(vec3 in);
  void backPropM2S(vec2 d, vec3 w, vec3 max, double eta); // muli-layer to single-layer
  void backPropS2M(vec3 d, vec2 w, vec3 max, double eta); // single-layer to multi-layer

  const vec3 getActivations() const;
  const vec3 getDelta() const;
  const vec2 getWeights() const;

};
#endif
