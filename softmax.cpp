#include "softmax.hpp"

SOFTMAX::SOFTMAX() {}

SOFTMAX::SOFTMAX(int batchSize, int input, int output)
{
  this -> batchSize = batchSize;
  this -> input = input;
  this -> output = output;

  intializeWs();
  intializeBs();
}

void SOFTMAX::intializeWs()
{
  Ws.resize(output);
  for(int i = 0; i < output; i++)
  {
    Ws.at(i).resize(input, 0);
  }
}

void SOFTMAX::intializeBs()
{
  Bs.resize(output, 0);
}

void SOFTMAX::feed(vec2 in)
{
  double temp;
  inputAs = in;
  Zs.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    Zs.at(i).resize(output);
    for(int j = 0; j < output; j++)
    {
      temp = 0;
      for(int k = 0; k < input; k++)
      {
        temp += Ws.at(j).at(k)*in.at(i).at(k);
      }
      temp += Bs.at(j);
      Zs.at(i).at(j) = temp;
    }
  }
  softmax(Zs);
}

void SOFTMAX::backProp(vec2 Y, double eta)
{
  Ds.resize(batchSize); // Ds = dC/dZ for each neuron
  for(int i = 0; i < batchSize; i++)
  {
    Ds.at(i).resize(output);
    for(int j = 0; j < output; j++)
    {
      Ds.at(i).at(j) = (As.at(i).at(j) - Y.at(i).at(j));
      Bs.at(j) -= eta*Ds.at(i).at(j)/batchSize;
      for(int k = 0; k < input; k++)
      {
        Ws.at(j).at(k) -= eta*inputAs.at(i).at(k)*Ds.at(i).at(j)/batchSize;
      }
    }
  }
}

void SOFTMAX::softmax(vec2 Zs)
{
  double temp;
  As.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    As.at(i).resize(output);
    for(int j = 0; j < output; j++)
    {
      temp = 0;
      for(int k = 0; k < output; k++)
      {
        temp += std::exp(Zs.at(i).at(k));
      }
      temp = std::exp(Zs.at(i).at(j)) / temp;
      As.at(i).at(j) = temp;
    }
  }
}

const vec2 SOFTMAX::getAs() const
{
  return As;
}

const vec2 SOFTMAX::getZs() const
{
  return Zs;
}

const vec1 SOFTMAX::getBs() const
{
  return Bs;
}

const vec2 SOFTMAX::getWs() const
{
  return Ws;
}

const vec2 SOFTMAX::getDs() const
{
  return Ds;
}
