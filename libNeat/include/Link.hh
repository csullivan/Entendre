#pragma once
#include <memory>

class Node;

class Link {
public:
  std::shared_ptr<Node> input;
  std::shared_ptr<Node> output;
  bool isRecurrent;
  bool isDelayed;

}
