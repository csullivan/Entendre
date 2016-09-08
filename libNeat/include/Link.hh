#pragma once
#include <memory>

class Node;

class Link {
public:
  Link(std::shared_ptr<Node>&,std::shared_ptr<Node>&);
  std::shared_ptr<Node> origin;
  std::shared_ptr<Node> destination;
  bool isRecurrent;
  bool isDelayed;

};
