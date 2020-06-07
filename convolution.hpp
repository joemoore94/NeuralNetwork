#ifndef _CONVOLUTION_HPP
#define _CONVOLUTION_HPP

#include <vector>
#include <cstdlib>
#include <iostream>
#include <random>

class CONVOLUTION
{
private:
  int batchSize;
  int imgX;
  int imgY;
  int conX;
  int conY;
  int layIn;
  int layOut;

  std::vector<double> dummy;
  std::vector<std::vector<double> > dummy2d;
  std::vector<std::vector<double> > biases;
  std::vector<std::vector<std::vector<double> > > weights;
  std::vector<std::vector<std::vector<double> > > Zs;
  std::vector<std::vector<std::vector<double> > > activations;

  double ReLU(double z);
  void intializeBiases();
  void intializeWeights();

public:
  CONVOLUTION(); // defualt constructor
  CONVOLUTION(int batchSize, int imgX, int imgY, int conX, int conY, int layIn, int layOut);

  void feed(std::vector<std::vector<double> > input);
  void feed(std::vector<std::vector<std::vector<double> > > input);
  std::vector<std::vector<std::vector<double> > > getActivations() const;


  void print2dVectors(std::vector<std::vector<double> > vec);
  void print3dVectors(std::vector<std::vector<std::vector<double> > > vec);

};
#endif
