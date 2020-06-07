#include "network.hpp"
#include "readfile.hpp"
#include "convolution.hpp"
#include "maxpool.hpp"

int main(int argc, char** argv)
{
  READFILE rf;
  CONVOLUTION con1(10,28,28,5,5,1,20); // (batchSize,imgX,imgY,conX,conY,layIn,layOut)
  MAXPOOL max1(10, 24, 24, );  // (batchSize,imgX,imgY,conX,conY,layIn,layOut)
  CONVOLUTION con2(10,24,24,5,5,20,20); // (batchSize,imgX,imgY,conX,conY,layIn,layOut)
  NETWORK net(10, rf, con1, con2); // (batchSize, ...)

  return EXIT_SUCCESS;
}
