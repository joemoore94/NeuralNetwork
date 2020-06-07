#include "readfile.hpp"

READFILE::READFILE()
{
  importTrainData("mnist_data/mnist_train.dat");
  importTestData("mnist_data/mnist_test.dat");
}

READFILE::READFILE(std::string trainfile, std::string testfile)
{
  importTrainData(trainfile);
  importTestData(testfile);
}

void READFILE::importTrainData(std::string trainfile)
{
  double temp;
  std::ifstream dataIn(trainfile);
  if(!dataIn.is_open())
		std::cout << "Error while opening the trainfile" << std::endl;
  for(int i=0; i<num_train_sets; i++)
  {
    dataIn >> temp;
    Y_train.push_back(vectorize(temp));
    dummy.clear();
    for(int j=0; j<num_inputs; j++)
    {
      dataIn >> temp;
      dummy.push_back(temp);
    }
    X_train.push_back(dummy);
  }
}

void READFILE::importTestData(std::string testfile)
{
  int temp;
  std::ifstream dataIn(testfile);
  if(!dataIn.is_open())
    std::cout << "Error while opening the testfile" << std::endl;
  for(int i=0; i<num_test_sets; i++)
  {
    dataIn >> temp;
    Y_test.push_back(vectorize(temp));
    dummy.clear();
    for(int j=0; j<num_inputs; j++)
    {
      dataIn >> temp;
      dummy.push_back((double)temp);
    }
    X_test.push_back(dummy);
  }
}

std::vector<double> READFILE::vectorize(int num)
{
  //std::cout << num << std::endl;
  dummy.clear();
  for(int i=0; i<num_outputs; i++)
  {
    if(i == num)
    {
      dummy.push_back(1.0);
    }
    else {dummy.push_back(0.0);}
    //std::cout << i << std::endl;
  }
  return dummy;
}

const std::vector<double> READFILE::getXtrain(int i) const {return X_train.at(i);}
const std::vector<double> READFILE::getYtrain(int i) const {return Y_train.at(i);}
const std::vector<double> READFILE::getXtest(int i) const {return X_test.at(i);}
const std::vector<double> READFILE::getYtest(int i) const {return Y_test.at(i);}
