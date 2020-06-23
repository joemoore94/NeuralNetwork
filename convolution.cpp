#include "convolution.hpp"

CONVOLUTION::CONVOLUTION() {}

CONVOLUTION::CONVOLUTION(int BS, int imgX, int imgY, int conX, int conY, int layers)
{
  this -> batchSize = BS;
  this -> imgX = imgX;
  this -> imgY = imgY;
  this -> conX = conX;
  this -> conY = conY;
  this -> layers = layers;

  intializeBiases();
  intializeWeights();
}

void CONVOLUTION::intializeBiases()
{
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0,1.0);

  biases.resize(layers);
  for(int i = 0; i < layers; i++)
  {
    biases.at(i) = dist(gen);
  }
}

void CONVOLUTION::intializeWeights()
{
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0,1.0);

  weights.resize(layers);
  for(int i = 0; i < layers; i++)
  {
    weights.at(i).resize(conX*conY);
    for(int j = 0; j < conX*conY; j++)
    {
      weights.at(i).at(j) = dist(gen);
    }
  }
}

void CONVOLUTION::feed(vec2 in)
{
  double temp;
  int outY = imgY-conY+1;
  int outX = imgX-conX+1;
  input_Activations = in;
  activations.resize(batchSize);
  Zs.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    activations.at(i).resize(layers);
    Zs.at(i).resize(layers);
    for (int h = 0; h < layers; h++)
    {
      activations.at(i).at(h).resize(outY*outX);
      Zs.at(i).at(h).resize(outY*outX);
      for(int j = 0; j < outY; j++)
      {
        for(int k = 0; k < outX; k++)
        {
          temp = 0;
          for(int l = 0; l < conY; l++)
          {
            for(int m = 0; m < conX; m++)
            {
              temp += in.at(i).at(j*imgX + k + l*imgX + m)*weights.at(h).at(m);
            }
          }
          temp += biases.at(h);
          Zs.at(i).at(h).at(j*outX + k) = temp;
          activations.at(i).at(h).at(j*outX + k) = ReLU(temp);
        }
      }
    }
  }
  //print3dVectors(activations);
}

void CONVOLUTION::feed(vec3 in)
{
  double temp;
  int outY = imgY-conY+1;
  int outX = imgX-conX+1;
  input_activations = in;
  activations.resize(batchSize);
  Zs.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
    activations.at(i).resize(layers);
    Zs.at(i).resize(layers);
    for (int h = 0; h < layers; h++)
    {
      activations.at(i).at(h).resize(outY*outX);
      Zs.at(i).at(h).resize(outY*outX);
      for(int j = 0; j < outY; j++)
      {
        for(int k = 0; k < outX; k++)
        {
          temp = 0;
          for(int l = 0; l < conY; l++)
          {
            for(int m = 0; m < conX; m++)
            {
              temp += in.at(i).at(h).at(j*imgX + k + l*imgX + m)*weights.at(h).at(m);
            }
          }
          temp += biases.at(h);
          Zs.at(i).at(h).at(j*outX + k) = temp;
          activations.at(i).at(h).at(j*outX + k) = ReLU(temp);
        }
      }
    }
  }
  //std::cout << input_activations.size() << '\n';
}

void CONVOLUTION::backPropM2S(vec2 d, vec3 w, vec3 max, double eta)
{
  // muli-layer to single-layer
  vec3& a = input_activations;
  int output = (imgY-conY+1)*(imgX-conX+1);
  delta.resize(batchSize);
  for (int i = 0; i < batchSize; i++)
  {
    delta.at(i).resize(layers);
    for (int j = 0; j < layers; j++)
    {
      delta.at(i).at(j).resize(output);
      for (int k = 0; k < output; k++)
      {
        for(int l = 0; l < (int)w.at(0).at(0).size(); l++)
        {
          for(int m = 0; m < (int)w.size(); m++)
          {
            if(max.at(i).at(j).at(k) == 1)
            {
              delta.at(i).at(j).at(k) += w.at(m).at(j).at(l)*d.at(i).at(m);
            }
          }
        }
        delta.at(i).at(j).at(k) *= ReLUP(Zs.at(i).at(j).at(k));
        biases.at(j) -= eta*delta.at(i).at(j).at(k);
        for(int n = 0; n < conX*conY; n++)
        {
          for(int p = 0; p < imgX*imgY; p++)
          {
            weights.at(j).at(n) -= eta*delta.at(i).at(j).at(k)*a.at(i).at(j).at(p)/batchSize;
          }
        }
      }
    }
  }
  //std::cout << a.at(0).at(0).size() << imgX*imgY << '\n';
}

void CONVOLUTION::backPropS2M(vec3 d, vec2 w, vec3 max, double eta)
{
  // single-layer to multi-layer
  vec2& a = input_Activations;
  int output = (imgY-conY+1)*(imgX-conX+1);
  delta.resize(batchSize);
  for (int i = 0; i < batchSize; i++)
  {
    delta.at(i).resize(layers);
    for (int j = 0; j < layers; j++)
    {
      delta.at(i).at(j).resize(output);
      for (int k = 0; k < output; k++)
      {
        for(int m = 0; m < (int)w.size(); m++)
        {
          if(max.at(i).at(j).at(k) == 1)
          {
            delta.at(i).at(j).at(k) += w.at(m).at(j)*d.at(i).at(j).at(m);
          }
        }
        delta.at(i).at(j).at(k) *= ReLUP(Zs.at(i).at(j).at(k));
        biases.at(j) -= eta*delta.at(i).at(j).at(k);
        for(int n = 0; n < conX*conY; n++)
        {
          weights.at(j).at(n) -= eta*delta.at(i).at(j).at(k)*a.at(i).at(j)/batchSize;
        }
      }
    }
  }
  //std::cout << max.at(0).at(0).size() << output << '\n';
}

double CONVOLUTION::ReLU(double z)
{
  if(z <= 0) {return 0;}
  else {return z;}
}

double CONVOLUTION::ReLUP(double z)
{
  if(z <= 0) {return 0;}
  else {return 1;}
}

const vec3 CONVOLUTION::getActivations() const
{
  return activations;
}

const vec3 CONVOLUTION::getDelta() const
{
  return delta;
}

const vec2 CONVOLUTION::getWeights() const
{
  return weights;
}
