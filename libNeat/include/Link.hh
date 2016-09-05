#pragma once
#include <memory>

class Node;

class Link {
public:
  Link(std::shared_ptr<Node>& in,std::shared_ptr<Node>& out);
  std::shared_ptr<Node> input;
  std::shared_ptr<Node> output;
  bool isRecurrent;
  bool isDelayed;

};
