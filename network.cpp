#include "network.hpp"

NETWORK::NETWORK() {}

NETWORK::NETWORK(int batchSize)
{
  this -> batchSize = batchSize;

  CONVOLUTION con1(batchSize,28,28,5,5,20);  // (batchSize,imgX,imgY,conX,conY,layers)
  MAXPOOL max1(batchSize,24,24,2,2,20);  // (batchSize,imgX,imgY,poolX,poolY,layers)
  CONVOLUTION con2(batchSize,12,12,5,5,20); // (batchSize,imgX,imgY,conX,conY,layers)
  MAXPOOL max2(batchSize,8,8,2,2,20); // (batchSize,imgX,imgY,poolX,poolY,layers)
  SIGMOID sig(batchSize,4,4,100,20); // (batchSize,imgX,imgY,output,layers) **fully connected layer**
  SOFTMAX sof(batchSize,100,10); // (batchSize,input,output)

  this -> con1 = con1;
  this -> max1 = max1;
  this -> con2 = con2;
  this -> max2 = max2;
  this -> sig = sig;
  this -> sof = sof;

  for(int i = 0; i < 5; i++) {SDG();}
  print3dVectors(con1.getActivations());
  //print2dVectors(Y);
}

void NETWORK::SDG()
{
  getbatch();
  feedFoward();
  backPropagation();
}

void NETWORK::feedFoward()
{
  con1.feed(X);
  max1.feed(con1.getActivations());
  con2.feed(max1.getActivations());
  max2.feed(con2.getActivations());
  sig.feed(max2.getActivations());
  sof.feed(sig.getActivations());
  //calculateCost(sof.getActivations());
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

void NETWORK::backPropagation()
{
  sof.backProp(Y, eta);
  sig.backProp(sof.getDelta(), sof.getWeights(), eta);
  con2.backPropM2S(sig.getDelta(), sig.getWeights(), max2.getMaxInput(), eta);
  con1.backPropS2M(con2.getDelta(), con2.getWeights(), max1.getMaxInput(), eta);
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
