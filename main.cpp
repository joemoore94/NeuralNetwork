#include "network.hpp"
#include "readfile.hpp"
#include "convolution.hpp"
#include "maxpool.hpp"
#include "sigmoid.hpp"
#include "softmax.hpp"

int main(int argc, char** argv)
{
  READFILE rf;
  CONVOLUTION con1(10,28,28,5,5,20); // (batchSize,imgX,imgY,conX,conY,layers)
  MAXPOOL max1(10,24,24,2,2,20);  // (batchSize,imgX,imgY,poolX,poolY,layers)
  CONVOLUTION con2(10,12,12,5,5,20); // (batchSize,imgX,imgY,conX,conY,layers)
  MAXPOOL max2(10,8,8,2,2,20);  // (batchSize,imgX,imgY,poolX,poolY,layers)
  SIGMOID sig(10,4,4,100,20); // (batchSize,imgX,imgY,output,layers) **fully connect all layers**
  SOFTMAX sof(10,100,10); // (batchSize,input,output)
  NETWORK net(10, rf, con1, max1, con2, max2, sig, sof); // (batchSize, ...)

  return EXIT_SUCCESS;
}
