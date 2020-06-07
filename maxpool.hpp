#ifndef _MAXPOOL_HPP
#define _MAXPOOL_HPP

#include <vector>
#include <cstdlib>
#include <iostream>

class MAXPOOL
{
private:
  int batchSize;
  int imgX;
  int imgY;
  int poolX;
  int poolY;
  int layers;

  std::vector<double> dummy;
  std::vector<std::vector<double> > dummy2d;
  std::vector<std::vector<std::vector<double> > > activations;

public:
  MAXPOOL(int BS, int imgX, int imgY, int poolX, int poolY, int layers);

  void feed(std::vector<std::vector<std::vector<double> > > input);

};
#endif
