#include "maxpool.hpp"

MAXPOOL::MAXPOOL(int BS, int imgX, int imgY, int poolX, int poolY, int layers)
{
  this -> batchSize = BS;
  this -> imgX = imgX;
  this -> imgY = imgY;
  this -> poolX = poolX;
  this -> poolY = poolY;
  this -> layers = layers;
}

void MAXPOOL::feed(std::vector<std::vector<std::vector<double> > > input)
{
  double temp, max;
  for (int i = 0; i < batchSize; i++)
  {
    dummy2d.clear();
    for (int j = 0; j < layers; j++)
    {
      dummy.clear();
      for (int k = 0; k < imgY; k+=poolY)
      {
        for (int l = 0; l < imgX; l+=poolX)
        {
          max = 0;
          for (int m = 0; m < poolY; m++)
          {
            for (int n = 0; n < poolX; n++)
            {
              temp = input.at(i).at(j).at(k*imgX + l + m*imgX + n);
              if(temp > max) {max = temp;}
            }
          }
          dummy.push_back(max);
        }
      }
      dummy2d.push_back(dummy);
    }
    activations.push_back(dummy2d);
  }
}

std::vector<std::vector<std::vector<double> > > MAXPOOL::getActivations() const
{
  return activations;
}
