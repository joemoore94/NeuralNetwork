#include "network.hpp"

NETWORK::NETWORK() {}

NETWORK::NETWORK(int batchSize, double eta)
{
  this -> batchSize = batchSize;
  this -> eta = eta;

  // (batchSize,input,output)
  SIGMOID sig1(batchSize,784,100);
  this -> sig1 = sig1;
  // (batchSize,input,output)
  SIGMOID sig2(batchSize,100,100);
  this -> sig2 = sig2;
  // (batchSize,input,output)
  SOFTMAX sof(batchSize,100,10);
  this -> sof = sof;

  for(int i = 0; i < 20; i++)
  {
    for(int j = 0; j < 100; j++)
    {
      SDG();
    }
    std::cout << i << ": ";
    test();
  }

}

void NETWORK::SDG()
{
  getbatch();
  feedFoward();
  backPropagation();
}

void NETWORK::feedFoward()
{
  sig1.feed(X);
  sig2.feed(sig1.getAs());
  sof.feed(sig2.getAs());
}

void NETWORK::backPropagation()
{
  sof.backProp(Y, eta);
  sig2.backProp(sof.getDs(), sof.getWs(), eta);
  sig1.backProp(sig2.getDs(), sig2.getWs(), eta);
}

void NETWORK::test()
{
  int numRight = 0;
  int numTest = rf.getNumTest();
  for(int i = 0; i < numTest; i += batchSize)
  {
    for(int j = 0; j < batchSize; j++)
    {
      X.at(j) = rf.getXtrain(i+j);
      Y.at(j) = rf.getYtrain(i+j);
    }
    feedFoward();
    for(int j = 0; j < batchSize; j++)
    {
      for(int k = 0; k < 10; k++)
      {
        if((sof.getAs().at(j).at(k) > 0.5) && (Y.at(j).at(k) == 1))
        {
          numRight += 1;
        }
      }
    }
  }
  std::cout << numRight << "/" << numTest << '\n';
}

void NETWORK::getbatch()
{
  typedef std::chrono::high_resolution_clock myclock;
  myclock::time_point beginning = myclock::now();
  myclock::duration d = myclock::now() - beginning;
  unsigned seed = d.count();

  std::default_random_engine gen(seed);
  std::uniform_int_distribution<int> dist(0, 60000);
  int n;
  X.resize(batchSize);
  Y.resize(batchSize);
  for (int i = 0; i < batchSize; i++)
  {
    n = dist(gen);
    X.at(i) = rf.getXtrain(n);
    Y.at(i) = rf.getYtrain(n);
    //std::cout << n << '\n';
  }
}

void NETWORK::calculateCost(vec2 in)
{
  double cost = 0;
  for(int i = 0; i < batchSize; i++)
  {
    for(int j = 0; j < (int)in.at(i).size(); j++)
    {
      if(Y.at(i).at(j) == 1)
      {
        cost -= log(in.at(i).at(j));
      }
    }
  }
  this -> cost = cost;
}






// --------------------------- print ------------------------------

void NETWORK::print1dVectors(vec1 vec)
{
  std::cout << "[ ";
  for(int i = 0; i < (int)(vec.size()); i++)
  {
    std::cout << vec.at(i) << " ";
  }
  std::cout << "]" << std::endl;
}

void NETWORK::print2dVectors(vec2 vec)
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

void NETWORK::print3dVectors(vec3 vec)
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
