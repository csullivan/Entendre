#include "Node.hh"

Node::Node(Type t, Function f) : type(t), functype(f) {

}

Node::Node(const std::shared_ptr<Node>& n) {
  activation = n->activation;
  type = Type(*n);
  functype = Function(*n);
  input_links = n->input_links;
  output_links = n->output_links;
}
