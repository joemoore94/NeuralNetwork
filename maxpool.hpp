#ifndef _MAXPOOL_HPP
#define _MAXPOOL_HPP

#include <vector>
#include <cstdlib>
#include <iostream>

using vec1 = std::vector<double>;
using vec2 = std::vector<std::vector<double> >;
using vec3 = std::vector<std::vector<std::vector<double> > >;

class MAXPOOL
{
private:
  int batchSize;
  int imgX;
  int imgY;
  int poolX;
  int poolY;
  int layers;

  vec3 max_input;
  vec3 activations;

public:
  MAXPOOL();
  MAXPOOL(int BS, int imgX, int imgY, int poolX, int poolY, int layers);

  void feed(vec3 in);

  const vec3 getActivations() const;
  const vec3 getMaxInput() const;
};
#endif
