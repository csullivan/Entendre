#include "Link.hh"
#include "Node.hh"

Link::Link(std::shared_ptr<Node>& in,
           std::shared_ptr<Node>& out)
  : input(in), output(out), isRecurrent(false), isDelayed(false) {

  // fill the participant nodes link lists
  in->output_links.push_back(std::make_shared<Link>(*this));
  out->input_links.push_back(std::make_shared<Link>(*this));

}
