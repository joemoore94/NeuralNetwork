#include "convolution.hpp"

CONVOLUTION::CONVOLUTION() // defualt constructor
{

}

CONVOLUTION::CONVOLUTION(int BS, int imgX, int imgY, int conX, int conY, int layIn, int layOut)
{
  this -> batchSize = BS;
  this -> imgX = imgX;
  this -> imgY = imgY;
  this -> conX = conX;
  this -> conY = conY;
  this -> layIn = layIn;
  this -> layOut = layOut;

  intializeBiases();
  intializeWeights();
}

void CONVOLUTION::intializeBiases()
{

  for(int i = 0; i < batchSize; i++)
  {
    dummy.clear();
    for(int j = 0; j < layOut; j++)
    {
      dummy.push_back(0);
    }
    biases.push_back(dummy);
  }
  //print2dVectors(biases);
}

void CONVOLUTION::intializeWeights()
{
  for(int i = 0; i < batchSize; i++)
  {
    dummy2d.clear();
    for(int j = 0; j < layOut; j++)
    {
      dummy.clear();
      for(int k = 0; k < conX*conY; k++)
      {
        dummy.push_back(0);
      }
      dummy2d.push_back(dummy);
    }
    weights.push_back(dummy2d);
  }
  //print3dVectors(weights);
}

void CONVOLUTION::feed(std::vector<std::vector<double> > input)
{
  double temp;
  activations.resize(batchSize);
  Zs.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    activations.at(i).resize(layOut);
    Zs.at(i).resize(layOut);
    for (int h = 0; h < layOut; h++)
    {
      for(int j = 0; j < imgY-conY+1; j++)
      {
        for(int k = 0; k < imgX-conX+1; k++)
        {
          temp = 0;
          for(int l = 0; l < conY; l++)
          {
            for(int m = 0; m < conX; m++)
            {
              temp += input.at(i).at(j*imgX + k + l*imgX + m)*weights.at(i).at(h).at(m);
            }
          }
          temp += biases.at(i).at(h);
          Zs.at(i).at(h).push_back(temp);
          activations.at(i).at(h).push_back(ReLU(temp));
        }
      }
    }
  }
  //print3dVectors(activations);
}

void CONVOLUTION::feed(std::vector<std::vector<std::vector<double> > > input)
{
  double temp;
  activations.resize(batchSize);
  Zs.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    activations.at(i).resize(layIn);
    Zs.at(i).resize(layIn);
    for (int h = 0; h < layIn; h++)
    {
      for(int j = 0; j < imgY-conY+1; j++)
      {
        for(int k = 0; k < imgX-conX+1; k++)
        {
          temp = 0;
          for(int l = 0; l < conY; l++)
          {
            for(int m = 0; m < conX; m++)
            {
              temp += input.at(i).at(h).at(j*imgX + k + l*imgX + m)*weights.at(i).at(h).at(m);
            }
          }
          temp += biases.at(i).at(h);
          Zs.at(i).at(h).push_back(temp);
          activations.at(i).at(h).push_back(ReLU(temp));
        }
      }
    }
  }
  //print3dVectors(activations);
}

double CONVOLUTION::ReLU(double z)
{
  if(z <= 0) {return 0;}
  else {return z;}
}

std::vector<std::vector<std::vector<double> > > CONVOLUTION::getActivations() const
{
  return activations;
}













// --------------------------- print ------------------------------

void CONVOLUTION::print2dVectors(std::vector<std::vector<double> > vec)
{
  for(int j = 0; j < (int)(vec.size()); j++)
  {
    std::cout << "[ ";
    for(int i = 0; i < (int)(vec[j].size()); i++)
    {
      std::cout << vec.at(j).at(i) << " ";
    }
    std::cout << "]" << std::endl << std::endl;
  }
}

void CONVOLUTION::print3dVectors(std::vector<std::vector<std::vector<double> > > vec)
{
  for(int k = 0; k < (int)(vec.size()); k++)
  {
    std::cout << "[";
    for(int j = 0; j < (int)(vec[k].size()); j++)
    {
      std::cout << "[ ";
      for(int i = 0; i < (int)(vec[k][j].size()); i++)
      {
        std::cout << vec.at(k).at(j).at(i) << " ";
      }
      if(j<(int)(vec[k].size())-1)
        std::cout << "]" << std::endl;
      else
        std::cout << "]";
    }
    std::cout << "]" << std::endl << std::endl;
  }
}
