#include "network.hpp"

NETWORK::NETWORK(int BS, READFILE& r, CONVOLUTION& c1, MAXPOOL& m1, CONVOLUTION& c2, MAXPOOL& m2,
  SIGMOID& s, SOFTMAX& sf) : rf(r), con1(c1), max1(m1), con2(c2), max2(m2), sig(s), sof(sf)
{
  this -> batchSize = BS;
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
  max1.feed(con1.getActivations());
  con2.feed(max1.getActivations());
  max2.feed(con2.getActivations());
  sig.feed(max2.getActivations());
  sof.feed(sig.getActivations());
  //print2dVectors(sig.getActivations());

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
