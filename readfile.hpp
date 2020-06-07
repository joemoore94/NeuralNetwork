#ifndef _READFILE_HPP
#define _READFILE_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>

class READFILE
{
private:
  int num_inputs = 784;
  int num_outputs = 10;
  int num_train_sets = 60000;
  int num_test_sets = 10000;
  std::vector<double> dummy;
  std::vector<std::vector<double> > X_train;
  std::vector<std::vector<double> > Y_train;
  std::vector<std::vector<double> > X_test;
  std::vector<std::vector<double> > Y_test;

  void importTrainData(std::string trainfile);
  void importTestData(std::string testfile);
  std::vector<double> vectorize(int num);

public:
  READFILE(); // default constructor
  READFILE(std::string trainfile, std::string testfile);
  const std::vector<double> getXtrain(int i) const;
  const std::vector<double> getYtrain(int i) const;
  const std::vector<double> getXtest(int i) const;
  const std::vector<double> getYtest(int i) const;

};
#endif
