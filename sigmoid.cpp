#include "sigmoid.hpp"

SIGMOID::SIGMOID() {}

SIGMOID::SIGMOID(int batchSize, int input, int output)
{
  this -> batchSize = batchSize;
  this -> input = input;
  this -> output = output;

  intializeWs();
  intializeBs();
}

void SIGMOID::intializeBs()
{
  typedef std::chrono::high_resolution_clock myclock;
  myclock::time_point beginning = myclock::now();
  myclock::duration d = myclock::now() - beginning;
  unsigned seed = d.count();

  std::default_random_engine gen(seed);
  std::normal_distribution<double> dist(0.0,1.0);
  int n;

  Bs.resize(output);
  for (int i = 0; i < output; i++)
  {
    n = dist(gen);
    Bs.at(i) = n;
  }
}

void SIGMOID::intializeWs()
{
  typedef std::chrono::high_resolution_clock myclock;
  myclock::time_point beginning = myclock::now();
  myclock::duration d = myclock::now() - beginning;
  unsigned seed = d.count();

  std::default_random_engine gen(seed);
  std::normal_distribution<double> dist(0.0,1.0);
  int n;

  Ws.resize(output);
  for(int i = 0; i < output; i++)
  {
    Ws.at(i).resize(input);
    for(int j = 0; j < input; j++)
    {
      n = dist(gen);
      Ws.at(i).at(j) = n;
    }
  }
}


void SIGMOID::feed(vec2 in)
{
  inputAs = in;
  double temp;
  Zs.resize(batchSize);
  As.resize(batchSize);
  for (int i = 0; i < batchSize; i++)
  {
    Zs.at(i).resize(output);
    As.at(i).resize(output);
    for (int j = 0; j < output; j++)
    {
      temp = 0;
      for (int k = 0; k < input; k++)
      {
        temp += Ws.at(j).at(k)*in.at(i).at(k);
      }
      temp += Bs.at(j);
      Zs.at(i).at(j) = temp;
      As.at(i).at(j) = sigmoid(temp);
    }
  }
}

void SIGMOID::backProp(vec2 d, vec2 w, double eta)
{
  int nextOutput = (int)w.size();
  // inputAs is the previous activations, d is the preceding deltas and w is the preceding weights
  Ds.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    Ds.at(i).resize(output);
    for(int j = 0; j < output; j++)
    {
      for(int k = 0; k < nextOutput; k++)
      {
        Ds.at(i).at(j) += w.at(k).at(j)*d.at(i).at(k);
      }
      Ds.at(i).at(j) *= sigPrime(Zs.at(i).at(j));
      Bs.at(j) -= eta*Ds.at(i).at(j)/batchSize;
      for(int m = 0; m < input; m++)
      {
        Ws.at(j).at(m) -= eta*Ds.at(i).at(j)*inputAs.at(i).at(m)/batchSize;
      }
    }
  }
}

double SIGMOID::sigmoid(double z)
{
  return 1 / (1 + std::exp(-z));
}

double SIGMOID::sigPrime(double z)
{
  return sigmoid(z)*(1-sigmoid(z));
}

const vec2 SIGMOID::getAs() const
{
  return As;
}

const vec2 SIGMOID::getDs() const
{
  return Ds;
}

const vec2 SIGMOID::getWs() const
{
  return Ws;
}
