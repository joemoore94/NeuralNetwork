#ifndef _NETWORK_HPP
#define _NETWORK_HPP

#include "convolution.hpp"
#include "readfile.hpp"
#include "maxpool.hpp"
#include "sigmoid.hpp"
#include "softmax.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cmath>

using vec1 = std::vector<double>;
using vec2 = std::vector<std::vector<double> >;
using vec3 = std::vector<std::vector<std::vector<double> > >;

class NETWORK
{
private:
  int batchSize = 10;
  double eta = 1.0;
  double lambda = 0.0;
  double cost;

  vec2 X;
  vec2 Y;

  void SDG();
  void feedFoward();
  void getbatch();
  void calculateCost(vec2 in);
  void backPropagation();

  READFILE rf;
  CONVOLUTION con1;
  MAXPOOL max1;
  CONVOLUTION con2;
  MAXPOOL max2;
  SIGMOID sig;
  SOFTMAX sof;
  
  void print1dVectors(vec1 vec);
  void print2dVectors(vec2 vec);
  void print3dVectors(vec3 vec);

public:
  NETWORK();
  NETWORK(int batchSize);

};
#endif
