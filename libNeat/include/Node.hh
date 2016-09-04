#pragma once
#include <memory>
#include <vector>

class Link;

class Node {
public:


  std::vector<std::shared_ptr<Link> > input_links;
  std::vector<std::shared_ptr<Link> > output_links;
  double activation;

  enum class Type {
    Sensor, Neuron
      };
  enum class Function {
    Bias, Input, Hidden, Output
      };

  Node() { ; }
  Node(Type,Function);

private:
  Type type;
  Function functype;
};
