#include "network.hpp"

NETWORK::NETWORK(int batchSize, READFILE& r, CONVOLUTION& c1, CONVOLUTION& c2)
: rf(r), con1(c1), con2(c2)
{
  this -> batchSize = batchSize;
  SDG();
}

void NETWORK::SDG()
{
  getbatch();
  feedFoward();
}

void NETWORK::feedFoward()
{
  con1.feed(X);
  con2.feed(con1.getActivations());
  
}

void NETWORK::getbatch()
{
  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist(0, 60000);
  int n;
  for (int i = 0; i < batchSize; i++)
  {
    n = dist(gen);
    X.push_back(rf.getXtrain(n));
    Y.push_back(rf.getYtrain(n));
  }
  //print2dVectors(X);
}
















// --------------------------- print ------------------------------

void NETWORK::print2dVectors(std::vector<std::vector<double> > vec)
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

void NETWORK::print3dVectors(std::vector<std::vector<std::vector<double> > > vec)
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
