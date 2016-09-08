#include "Link.hh"
#include "Node.hh"

Link::Link(std::shared_ptr<Node>& ori,
           std::shared_ptr<Node>& dest)
  : origin(ori), destination(dest), isRecurrent(false), isDelayed(false) {

}
