#include "network.hpp"
#include "readfile.hpp"
#include "convolution.hpp"
#include "maxpool.hpp"

int main(int argc, char** argv)
{
  READFILE rf;
  CONVOLUTION con1(10,28,28,5,5,20); // (batchSize,imgX,imgY,conX,conY,layers)
  MAXPOOL max1(10, 24, 24, 2, 2, 20);  // (batchSize,imgX,imgY,poolX,poolY,layers)
  CONVOLUTION con2(10,12,12,5,5,20); // (batchSize,imgX,imgY,conX,conY,layers)
  MAXPOOL max2(10, 8, 8, 2, 2, 20);  // (batchSize,imgX,imgY,poolX,poolY,layers)
  NETWORK net(10, rf, con1, max1, con2, max2); // (batchSize, ...)

  return EXIT_SUCCESS;
}
