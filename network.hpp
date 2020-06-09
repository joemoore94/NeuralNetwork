#ifndef _NETWORK_HPP
#define _NETWORK_HPP

#include "convolution.hpp"
#include "readfile.hpp"
#include "maxpool.hpp"
#include "sigmoid.hpp"
#include "softmax.hpp"
#include <vector>
#include <random>
#include <cstdlib>
#include <iostream>

class NETWORK
{
private:
  double eta = 0.0;
  double lambda = 0.0;
  int batchSize;

  std::vector<std::vector<double> > X;
  std::vector<std::vector<double> > Y;
  std::vector<std::vector<double> > biases;
  std::vector<std::vector<std::vector<double> > > weights;

  void feedFoward();
  void getbatch();

  READFILE& rf;
  CONVOLUTION& con1;
  MAXPOOL& max1;
  CONVOLUTION& con2;
  MAXPOOL& max2;
  SIGMOID& sig;
  SOFTMAX& sof;

  void print2dVectors(std::vector<std::vector<double> > vec);
  void print3dVectors(std::vector<std::vector<std::vector<double> > > vec);

public:
  NETWORK(int BS, READFILE& rf, CONVOLUTION& con1, MAXPOOL& max1, CONVOLUTION& con2,
    MAXPOOL& max2, SIGMOID& sig, SOFTMAX& sof);
  void SDG();

};
#endif
