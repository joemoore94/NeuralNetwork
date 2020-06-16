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

void MAXPOOL::feed(vec3 in)
{
  double temp, max;
  int maxIndex;
  max_input.resize(batchSize);
  activations.resize(batchSize);
  for (int i = 0; i < batchSize; i++)
  {
    max_input.at(i).resize(layers);
    activations.at(i).resize(layers);
    for (int j = 0; j < layers; j++)
    {
      max_input.at(i).at(j).resize(imgY*imgX);
      activations.at(i).at(j).resize(imgY*imgX);
      for (int k = 0; k < imgY; k+=poolY)
      {
        for (int l = 0; l < imgX; l+=poolX)
        {
          max = 0;
          maxIndex = 0;
          for (int m = 0; m < poolY; m++)
          {
            for (int n = 0; n < poolX; n++)
            {
              temp = in.at(i).at(j).at(k*imgX + l + m*imgX + n);
              if(temp >= max)
              {
                max = temp;
                maxIndex = k*imgX + l + m*imgX + n;
              }
            }
          }
          max_input.at(i).at(j).at(maxIndex) = 1;
          activations.at(i).at(j).at(k*imgY + l) = max;
        }
      }
    }
  }
}


const vec3 MAXPOOL::getActivations() const
{
  return activations;
}

const vec3 MAXPOOL::getMaxInput() const
{
  return max_input;
}
