#ifndef _SIGMOID_HPP
#define _SIGMOID_HPP

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

using vec1 = std::vector<double>;
using vec2 = std::vector<std::vector<double> >;
using vec3 = std::vector<std::vector<std::vector<double> > >;

class SIGMOID
{
private:
  int batchSize;
  int output;
  int input;

  vec1 Bs;
  vec2 Ws;
  vec2 Ds;
  vec2 Zs;
  vec2 As;
  vec2 inputAs;

  double sigmoid(double z);
  double sigPrime(double z);
  void intializeBs();
  void intializeWs();

public:
  SIGMOID();
  SIGMOID(int batchSize, int input, int output);

  void feed(vec2 in);
  void backProp(vec2 d, vec2 w, double eta);

  const vec2 getAs() const;
  const vec2 getDs() const;
  const vec2 getWs() const;

};
#endif
