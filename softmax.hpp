#ifndef _SOFTMAX_HPP
#define _SOFTMAX_HPP

#include <vector>
#include <cstdlib>
#include <iostream>
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

  vec1 Bs;
  vec2 Ws;
  vec2 Ds;
  vec2 Zs;
  vec2 As;
  vec2 inputAs;

  void intializeWs();
  void intializeBs();
  void softmax(vec2 Zs);

public:
  SOFTMAX();
  SOFTMAX(int batchSize, int input, int output);

  void feed(vec2 in);
  void backProp(vec2 out, double eta);


  const vec2 getAs() const;
  const vec2 getZs() const;
  const vec1 getBs() const;
  const vec2 getWs() const;
  const vec2 getDs() const;

};
#endif
